import os
import torch
from unsloth import FastModel  # Use same import as gemma_test.py
from unsloth import is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from peft import PeftModel, LoraConfig
from transformers import TrainerCallback, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_callback import TrainerState, TrainerControl

from callm.utils import Logger, log_model_wrapper_diagnostics, reset_forward_overrides
from callm.adapter_manager import AdapterManager

class ConsoleLogCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        try:
            step = int(getattr(state, "global_step", 0) or 0)
            epoch = getattr(state, "epoch", None)
            parts = [f"step={step}"]
            if epoch is not None:
                parts.append(f"epoch={epoch}")
            if isinstance(logs, dict):
                # Keep a few common keys first
                for k in ["loss", "eval_loss", "grad_norm", "learning_rate", "lr"]:
                    if k in logs:
                        parts.append(f"{k}={logs[k]}")
                # Append any other numeric logs
                for k, v in logs.items():
                    if k in {"loss", "eval_loss", "grad_norm", "learning_rate", "lr"}:
                        continue
                    try:
                        # Only print simple scalars
                        if isinstance(v, (int, float)):
                            parts.append(f"{k}={v}")
                    except Exception:
                        pass
            msg = "[trainer] " + " ".join(parts)
            Logger.instance().debug(msg)
        except Exception:
            # Never let logging crash training
            pass


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.max_seq_length = self.config["max_seq_length"]

        # Always load the base model
        Logger.instance().debug("Initializing base model...")
        self.model, self.tokenizer = self.load_base_model(self.config["base_model"])
        Logger.instance().debug("Base model loaded.")

        # Centralized adapter manager
        self.adapter_manager = AdapterManager()

    def load_base_model(self, model_name: str):
        Logger.instance().debug(f"Loading model from {model_name}...")
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.config["load_in_4bit"],
            load_in_8bit=False,  # Match gemma_test.py
            full_finetuning=False,  # Match gemma_test.py
            dtype=torch.bfloat16,  # Explicitly set model dtype to bfloat16
        )
        return model, tokenizer

    def load_lora(self, peft_dir: str):
        Logger.instance().debug("Loading LoRA model...")
        if os.path.exists(peft_dir):
            self.load_existing_lora(peft_dir)
        else:
            self.load_new_lora()
            Logger.instance().debug("Starting training from new LoRA.")

    def load_existing_lora(self, peft_dir: str):
        Logger.instance().debug(f"Loading adapter from {peft_dir}...")
        try:
            self.model = self.adapter_manager.attach_adapter(self.model, peft_dir)
            # Move inactive adapters to CPU  
            self.adapter_manager.set_active(self.model, "default", device_inactive="cpu")
            Logger.instance().debug("Checkpoint adapter loaded successfully.")
        except RuntimeError as e:
            Logger.instance().critical(f"Error loading checkpoint adapter: {e}")
            exit(1)
        except FileNotFoundError as e:
            Logger.instance().critical(f"Error loading checkpoint adapter - file not found: {e}")
            exit(1)

    def load_new_lora(self):
        Logger.instance().debug("Creating new LoRA model...")
        Logger.instance().debug(f"Base model name: {self.config['base_model']}")
        Logger.instance().debug(f"Model type: {type(self.model)}")
        try:
            self.model = self.adapter_manager.create_new_adapter(self.model, self.config)
            # Move inactive adapters to CPU, keep active on current device
            if isinstance(self.model, PeftModel):
                self.adapter_manager.set_active(self.model, "default", device_inactive="cpu")
            Logger.instance().debug("New LoRA model created.")
        except Exception as e:
            Logger.instance().critical(f"Error creating LoRA model: {e}")
            raise


    def train_lora(self, train_dataset, eval_dataset, lora_save_path):

        is_online = str(self.config.get("stream_type", "")).lower() == "online"
        
        # Disable wandb for online runs to avoid multiple run issue with newer TRL versions
        wandb_name = self.config.get("wandb_name")
        if is_online or not wandb_name or "none" in str(wandb_name).lower():
            report = "none"
        else:
            report = "wandb"

        # Reduce allocator fragmentation risk (needs to be set before CUDA init; best-effort here)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")

        # Wrapper mitigation is only needed for long-running online streams.
        if is_online:
            log_model_wrapper_diagnostics(self.model, label="train_lora.start.pre_reset")
            cleared = reset_forward_overrides(self.model)
            if cleared:
                Logger.instance().debug(f"[Diag][wrappers][train_lora.start] reset_forward_instance_attrs={cleared}")
            log_model_wrapper_diagnostics(self.model, label="train_lora.start.post_reset")

        # Use fixed batch size from config (no dynamic adjustment)
        train_bs = int(self.config.get("batch_size", 4))

        # Build trainer once with configured batch size
        args = SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=train_bs,
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            warmup_steps=max(1, int(self.config["num_epochs"] * 10)),  # Simple warmup calculation
            num_train_epochs=self.config["num_epochs"],
            learning_rate=self.config["learning_rate"],
            logging_steps=self.config["eval_step"],
            logging_strategy="steps",
            logging_first_step=True,
            optim="adamw_8bit",  # Use simpler optimizer like gemma_test.py
            weight_decay=self.config["weight_decay"],
            lr_scheduler_type=self.config.get("lr_scheduler_type", "constant"),  # Use config value
            seed=self.config["seed"],
            output_dir=lora_save_path,
            report_to=report,
            save_steps=self.config["save_step"],
            eval_steps=self.config["eval_step"],
            eval_strategy="steps",  # Match save strategy
            save_strategy="steps",  # Explicit save policy
            save_total_limit=1,
            load_best_model_at_end=self.config.get("load_best_model_at_end", False),
            metric_for_best_model="eval_loss" if self.config.get("load_best_model_at_end", False) else None,
            greater_is_better=False if self.config.get("load_best_model_at_end", False) else None,
            bf16=True,  # Match model precision
        )

        trainer = SFTTrainer(
            model=self.model,  # Use the loaded or newly created lora model
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=args,
            packing=False,
        )

        if is_online:
            # Diagnostics: see if trainer/accelerate initialization wrapped forward in-place
            log_model_wrapper_diagnostics(trainer.model, label="train_lora.after_trainer_init")
        # Mirror HF training logs to our Logger regardless of integrations
        try:
            trainer.add_callback(ConsoleLogCallback())
        except Exception:
            pass
        
        # Add early stopping callback if enabled
        if self.config.get("early_stopping", False):
            trainer.add_callback(EarlyStoppingCallback())

        # Start model training
        trainer.train()

        # Save the final model as LoRA adapters (only LoRA adapters, not full model)
        self.adapter_manager.save_adapter(trainer.model, trainer.tokenizer, lora_save_path)

        # Keep the trained in-memory model for subsequent chunks
        self.model = trainer.model

        # Always-on (online only): clear training-time forward wrappers so they do not
        # accumulate across chunks.
        if is_online:
            cleared_post = reset_forward_overrides(self.model)
            if cleared_post:
                Logger.instance().debug(f"[Diag][wrappers][train_lora.after_train] reset_forward_instance_attrs={cleared_post}")
            log_model_wrapper_diagnostics(self.model, label="train_lora.after_train.post_reset")
        
        # Clean up trainer and its accelerator to free memory
        try:
            if hasattr(trainer, 'accelerator'):
                trainer.accelerator = None
            del trainer
            Logger.instance().debug("Trainer cleanup completed")
        except Exception as e:
            Logger.instance().debug(f"Trainer cleanup warning: {e}")
