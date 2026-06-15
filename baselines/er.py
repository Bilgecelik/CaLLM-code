import os
import torch
import pandas as pd
import numpy as np
from typing import List

from trl import SFTTrainer
from unsloth import FastLanguageModel as FastModel, is_bfloat16_supported
from transformers import TrainingArguments, EarlyStoppingCallback
from callm.data.trace.prepare_trace_data import prepare_task_incremental_stream
from callm.lora_trainer import ModelTrainer
from callm.training_manager import TrainingManager
from callm.utils import Logger, HF_TOKEN


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def concat_inputs(input_ids, attention_mask, labels, buffer_input_ids, buffer_attention_mask, buffer_labels):
    device = input_ids.device
    input_ids = torch.cat((input_ids, buffer_input_ids.to(device)), dim=0)
    attention_mask = torch.cat((attention_mask, buffer_attention_mask.to(device)), dim=0)
    labels = torch.cat((labels, buffer_labels.to(device)), dim=0)
    return input_ids, attention_mask, labels


class Buffer:
    """
        Data buffer to store previously seen examples.
        Code inspired from https://github.com/which47/LLMCL/tree/main.

        IMPORTANT: Do not seed RNGs here. We rely on the global seeding policy
        in callm.utils.set_seed(seed, determinism_level) so ER matches CALLM's
        determinism_level behavior.
    """
    def __init__(self, buffer_size, device, pad_id, ignore_index=-100):
        self.buffer_size = buffer_size
        self.device = device
        self.pad_id = pad_id
        self.ignore_index = ignore_index

        self.num_seen_examples = 0
        self.attributes = ['input_ids', 'attention_mask', 'labels', 'logits', 'task_labels']
        self.init_buffer()

    def init_buffer(self):
        for attr_str in self.attributes:
            setattr(self, attr_str, [None for _ in range(self.buffer_size)])

    def add_data(self, input_ids, attention_mask=None, labels=None, logits=None, task_labels=None):
        n = input_ids.shape[0] if hasattr(input_ids, 'shape') else len(input_ids)
        for i in range(n):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.input_ids[index] = input_ids[i].detach().clone().to(self.device)
                if attention_mask is not None:
                    self.attention_mask[index] = attention_mask[i].detach().clone().to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].detach().clone().to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].detach().clone().to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].detach().clone().to(self.device)

    def get_data(self, size, pad_to):
        n = len(self.input_ids)
        if size > min(self.num_seen_examples, n):
            size = min(self.num_seen_examples, n)

        choice = np.random.choice(min(self.num_seen_examples, n), size=size, replace=False)
        if len(choice) == 0:
            return None, None
        # for left padding
        input_ids = []
        attention_mask = []
        labels = []

        for i in choice:
            if pad_to >= self.input_ids[i].shape[-1]:
                input_ids.append(torch.cat(
                    (torch.full((pad_to - self.input_ids[i].shape[-1],), self.pad_id, dtype=torch.long).to(self.device),
                     self.input_ids[i]), dim=-1)
                )
                if self.attention_mask[i] is not None:
                    attention_mask.append(torch.cat(
                        (torch.full((pad_to - self.attention_mask[i].shape[-1],), 0, dtype=torch.long).to(self.device),
                         self.attention_mask[i]), dim=-1)
                    )
                if self.labels[i] is not None:
                    labels.append(torch.cat(
                        (torch.full((pad_to - self.labels[i].shape[-1],), self.ignore_index, dtype=torch.long).to(self.device),
                         self.labels[i]), dim=-1)
                    )
            else:
                input_ids.append(self.input_ids[i][-pad_to:])
                if self.attention_mask[i] is not None:
                    attention_mask.append(self.attention_mask[i][-pad_to:])
                if self.labels[i] is not None:
                    labels.append(self.labels[i][-pad_to:])

        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        labels = torch.stack(labels)
        return input_ids, attention_mask, labels

    def is_empty(self):
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def empty(self):
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0

    def get_all_data(self):
        ret_tuple = (torch.stack([ee.cpu()
                                  for ee in self.input_ids]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple


class ERTrainer(SFTTrainer):
    def __init__(self, dataset_id, buffer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_id = dataset_id
        self.buffer = buffer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        if self.dataset_id == 0:
            self.buffer.add_data(inputs["input_ids"], inputs["attention_mask"], inputs["labels"])
            outputs = model(**inputs)
        else:
            buffer_inputs, buffer_attention_mask, buffer_labels = self.buffer.get_data(inputs["input_ids"].shape[0], inputs["input_ids"].shape[1])
            if buffer_inputs is not None and buffer_attention_mask is not None and buffer_labels is not None:
                inputs["input_ids"], inputs["attention_mask"], inputs["labels"] = concat_inputs(inputs["input_ids"], inputs["attention_mask"], inputs["labels"], buffer_inputs, buffer_attention_mask,
                                                                                                buffer_labels)
            outputs = model(**inputs)
            self.buffer.add_data(inputs["input_ids"], inputs["attention_mask"], inputs["labels"])

        return (outputs.loss, outputs) if return_outputs else outputs.loss


class ERModelTrainer(ModelTrainer):
    def __init__(self, config):
        super().__init__(config)

        self.load_new_lora()

        # Initialize replay buffer. RNG seeding is controlled globally via
        # callm.utils.set_seed(seed, determinism_level) in main.py.
        self.buffer = Buffer(
            self.config["buffer_size"],
            'cpu',
            pad_id=self.tokenizer.pad_token_id,
            ignore_index=-100,
        )

    def train_lora(self, dataset, dataset_id, lora_save_path):
        # Define training arguments (rest is the same)
        training_args = TrainingArguments(
            output_dir=self.config["output_dir"],
            num_train_epochs=self.config["num_epochs"],
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"],
            weight_decay=self.config["weight_decay"],
            learning_rate=self.config["learning_rate"],
            logging_dir="./logs",
            logging_steps=self.config["eval_step"],
            eval_strategy="steps",
            eval_steps=self.config["eval_step"],
            save_strategy="steps",
            save_steps=self.config["save_step"],
            save_total_limit=1,
            load_best_model_at_end=self.config.get("load_best_model_at_end", False),
            metric_for_best_model="eval_loss" if self.config.get("load_best_model_at_end", False) else None,
            greater_is_better=False,
            hub_token=HF_TOKEN,
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            optim=self.config["optimizer"],
            max_grad_norm=self.config["max_grad_norm"],
            warmup_steps=max(1, int(self.config["num_epochs"] * 10)),
            lr_scheduler_type=self.config["lr_scheduler_type"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            seed=self.config["seed"],
            report_to="none" if self.config.get("debug") or not self.config.get("wandb_name") else "wandb",
        )

        # Initialize the Trainer (rest is the same)
        trainer = ERTrainer(
            dataset_id=dataset_id,
            buffer=self.buffer,
            model=self.model,  # Use the loaded or newly created lora model
            tokenizer=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            args=training_args,
            packing=False,
        )
        
        # Add early stopping callback if enabled
        if self.config.get("early_stopping", False):
            trainer.add_callback(EarlyStoppingCallback())

        # Start model training
        trainer.train()

        # Save the final model as LoRA adapters (only LoRA adapters, not full model)
        trainer.model.save_pretrained(lora_save_path)
        trainer.tokenizer.save_pretrained(lora_save_path)
        Logger.instance().debug(f"LoRA weights saved to: {lora_save_path}")


class ER(TrainingManager):
    """
        A subclass of TrainingManager designed for training and evaluation of EWC on a trained LoRA.
        Continual learning with a single LoRA and EWC regularization.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
        self.trainer = ERModelTrainer(config)
        self.router = None

    def init_generator(self):
        # Reuse the trainer's model instance to save memory
        if self.generator.model is None:
            self.generator.model = self.trainer.model
            self.generator.tokenizer = self.trainer.tokenizer
            # Compile for inference only once
            try:
                self.generator.model = FastModel.for_inference(self.generator.model)
                Logger.instance().debug("Model compiled for inference")
            except Exception as e:
                Logger.instance().debug(f"Model compilation failed: {e}")
        
        # Ensure model is in eval mode
        self.generator.model.eval()

    def _calculate_backward_transfer(self, data_stream: List[dict], current_dataset_id: int):
        """Calculates backward transfer for all previous tasks."""
        bwt_results = {}
        bwt_sum = 0
        num_prev_tasks = 0

        Logger.instance().debug(f"==== BWT Calculation for Task {current_dataset_id} ====")

        for prev_task_id in range(current_dataset_id):
            prev_dataset = data_stream[prev_task_id]
            prev_dataset_name = self.config['datasets'][prev_task_id]

            Logger.instance().debug(f"- Previous Task {prev_task_id}: {prev_dataset_name}")
            prev_task_performance, _ = self._evaluate_task(prev_dataset, prev_task_id)
            Logger.instance().debug(f"  - Evaluated Performance: {prev_task_performance}")

            # Update past task performance and peft selection table
            Logger.instance().debug(f"  - Updating performance table at ({prev_dataset_name}, {current_dataset_id + 1}) with {prev_task_performance}")
            self.performance_table.loc[self.performance_table["Task"] == prev_dataset_name, str(current_dataset_id + 1)] = prev_task_performance  # Ensure col is str if necessary

            # Compute BWT with proper normalization
            if prev_task_id in self.task_performance_history:
                past_performance = self.task_performance_history[prev_task_id]
                Logger.instance().debug(f"  - Past recorded performance: {past_performance}")
                
                # Get metric type and scale for normalization
                metric_name = self.metrics[prev_task_id]
                scale = self.metric_scale.get(metric_name, 1.0)
                
                # Normalize both performances before calculating BWT
                past_norm = past_performance / scale
                current_norm = prev_task_performance / scale
                
                # BWT is the normalized difference
                bwt = current_norm - past_norm
                
                Logger.instance().debug(f"  - Past: {past_performance:.3f} -> {past_norm:.3f} (scale: {scale})")
                Logger.instance().debug(f"  - Current: {prev_task_performance:.3f} -> {current_norm:.3f} (scale: {scale})")
                Logger.instance().debug(f"  - Normalized BWT ({metric_name}): {bwt:.4f}")

                bwt_results[prev_task_id] = bwt
                bwt_sum += bwt
                num_prev_tasks += 1
            else:
                Logger.instance().warning(f"  - WARNING: No past performance found for {prev_dataset_name} in history!")

            # Compute average BWT - needs to change to normalized
        bwt_score = bwt_sum / num_prev_tasks if num_prev_tasks > 0 else 0
        Logger.instance().debug(f"Final BWT Score: {bwt_score}")
        return bwt_score, bwt_results

    def train_single_dataset(self, dataset: dict, dataset_id: int):
        """ Initialize a new LoRA and train the model with it. """
        # Free generator references before training to avoid duplicate VRAM usage
        import gc
        try:
            self.generator.model = None
            self.generator.tokenizer = None
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'ipc_collect'):
                    torch.cuda.ipc_collect()
        except Exception:
            pass
        
        self.trainer.train_lora(dataset, dataset_id, self.config["output_dir"])

    def train_er(self):
        """Trains the model in a continual learning setting and tracks performance over time."""
        # Log configuration for reproducibility
        Logger.instance().debug(f"ER Training Configuration:")
        Logger.instance().debug(f"  - seed: {self.config.get('seed', 0)}")
        Logger.instance().debug(f"  - gen_batch_size: {self.config.get('gen_batch_size', self.config.get('batch_size', 2))}")
        Logger.instance().debug(f"  - batch_size: {self.config.get('batch_size', 16)}")
        Logger.instance().debug(f"  - buffer_size: {self.config.get('buffer_size', 1000)}")
        
        data_stream = prepare_task_incremental_stream(self.config["data_stream"], self.config["datasets"],
                                                      self.trainer.tokenizer.eos_token, self.config)

        # Initialize performance tracking table
        task_names = self.config["datasets"]
        self.performance_table = pd.DataFrame(columns=["Task"] + [str(i) for i in range(1, len(data_stream) + 1)])
        self.performance_table["Task"] = task_names
        # Add placeholder rows for BWT and Average, ensuring they exist only once
        self.performance_table.loc[len(self.performance_table)] = ["BWT"] + [None] * len(data_stream)
        self.performance_table.loc[len(self.performance_table)] = ["Average"] + [None] * len(data_stream)

        for dataset_id, dataset in enumerate(data_stream):
            Logger.instance().debug(f"Start training with dataset {dataset_id}: {task_names[dataset_id]}")
            self.train_single_dataset(dataset, dataset_id)
            Logger.instance().debug(f"End training with dataset {dataset_id}")
            
            # Log memory usage after training
            from callm.memory import MemoryManager
            stats = MemoryManager.cleanup(label="training.end")
            if stats is not None:
                Logger.instance().debug(
                    f"GPU memory after cleanup: allocated {stats.allocated_gb:.2f} GB | reserved {stats.reserved_gb:.2f} GB"
                )

            # Evaluate model
            # train_single_dataset() clears generator.model/tokenizer to avoid duplicate VRAM usage.
            # Re-initialize generator before evaluation if needed.
            if self.generator.model is None or self.generator.tokenizer is None:
                self.init_generator()

            current_task_performance, _ = self._evaluate_task(dataset, dataset_id)
            Logger.instance().debug(f"Current task performance: {current_task_performance}")

            # Store performance in tracking history
            self.task_performance_history[dataset_id] = current_task_performance

            # Update the performance table and peft selection table
            self.performance_table.loc[self.performance_table["Task"] == task_names[dataset_id], str(dataset_id + 1)] = current_task_performance

            # Evaluate backward transfer (BWT)
            bwt_score, bwt_results = self._calculate_backward_transfer(data_stream, dataset_id)
            if bwt_results:
                Logger.instance().debug("Backward Transfer Results:")
                for prev_task_id, bwt in bwt_results.items():
                    Logger.instance().debug(f"  BWT({task_names[prev_task_id]}): {bwt:.4f}")
                Logger.instance().debug(f"  Aggregated BWT: {bwt_score:.4f}")

            # **Update existing BWT row instead of adding a new one**
            self.performance_table.loc[self.performance_table["Task"] == "BWT", str(dataset_id + 1)] = bwt_score

            # Calculate and update average performance with proper normalization
            # Get all task rows (excluding BWT and Average rows)
            task_rows = self.performance_table[
                ~self.performance_table["Task"].isin(["BWT", "Average"])
            ]
            
            # Calculate normalized average
            total_normalized = 0
            valid_tasks = 0
            
            for _, row in task_rows.iterrows():
                task_name = row["Task"]
                current_performance = row[str(dataset_id + 1)]
                
                if pd.notna(current_performance) and current_performance is not None:
                    # Find the dataset_id for this task to get the metric
                    task_idx = self.config["datasets"].index(task_name)
                    metric_name = self.metrics.get(task_idx, "accuracy")
                    scale = self.metric_scale.get(metric_name, 1.0)
                    
                    # Normalize the performance
                    normalized_performance = float(current_performance) / scale
                    total_normalized += normalized_performance
                    valid_tasks += 1
                    
                    Logger.instance().debug(f"  - Task {task_name}: {current_performance:.3f} -> {normalized_performance:.3f} (scale: {scale})")
            
            avg_performance = total_normalized / valid_tasks if valid_tasks > 0 else 0
            Logger.instance().debug(f"  - Normalized Average: {avg_performance:.4f}")
            self.performance_table.loc[self.performance_table["Task"] == "Average", str(dataset_id + 1)] = avg_performance

            Logger.instance().debug("Updated Performance Table:")
            Logger.instance().debug(self.performance_table.to_string(index=False))

        # Save tables
        self.performance_table.to_csv(os.path.join(self.config["output_dir"], "performance_table.csv"), index=False)
