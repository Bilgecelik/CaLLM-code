import os
import pandas as pd
from unsloth import FastLanguageModel as FastModel

from callm.data.trace.prepare_trace_data import prepare_task_incremental_stream
from callm.training_manager import TrainingManager
from callm.generator import Generator
from callm.utils import Logger


class InferenceOnly(TrainingManager):
    """
        A subclass of TrainingManager designed exclusively for inference on each single task.
        It omits training-related components and retains only the necessary attributes for evaluation and inference.
        Single-task (non continual) scenario.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.generator.model, self.generator.tokenizer = FastModel.from_pretrained(
            model_name=self.config["base_model"],
            max_seq_length=self.config["max_seq_length"],
            dtype=None,
            load_in_4bit=self.config["load_in_4bit"],
        )

        self.generator.model = FastModel.for_inference(self.generator.model)
        self.generator.model.eval()  # Ensure model is in evaluation mode

    def eval_single_task(self):
        # Use the generator's tokenizer which was properly loaded
        eos_token = self.generator.tokenizer.eos_token if self.generator.tokenizer else "</s>"
        data_stream = prepare_task_incremental_stream(self.config["data_stream"], self.config["datasets"],
                                                      eos_token, self.config)

        task_names = self.config["datasets"]
        
        # Debug: Check if model and tokenizer are properly loaded
        Logger.instance().debug(f"Generator model loaded: {self.generator.model is not None}")
        Logger.instance().debug(f"Generator tokenizer loaded: {self.generator.tokenizer is not None}")
        
        for dataset_id, dataset in enumerate(data_stream):
            Logger.instance().debug(f"Start evaluating with dataset {dataset_id}: {task_names[dataset_id]}")
            
            # Ensure generator has model and tokenizer (defensive programming)
            if self.generator.model is None or self.generator.tokenizer is None:
                Logger.instance().warning("Model or tokenizer not properly loaded, attempting to reload...")
                self.generator.model, self.generator.tokenizer = FastModel.from_pretrained(
                    model_name=self.config["base_model"],
                    max_seq_length=self.config["max_seq_length"],
                    dtype=None,
                    load_in_4bit=self.config["load_in_4bit"],
                )
                self.generator.model = FastModel.for_inference(self.generator.model)
                Logger.instance().debug("Model and tokenizer reloaded successfully")
            
            # Ensure model is in evaluation mode
            if hasattr(self.generator.model, 'eval'):
                self.generator.model.eval()
                Logger.instance().debug("Model set to evaluation mode")

            # Evaluate model
            current_task_performance, _ = self._evaluate_task(dataset, dataset_id)
            Logger.instance().debug(f"Current task performance: {current_task_performance}")

            # Store performance in tracking history
            self.task_performance_history[dataset_id] = current_task_performance

        performance_table = pd.DataFrame([self.task_performance_history])
        Logger.instance().debug("Task performance history:")
        Logger.instance().debug(performance_table.to_string(index=False))
        performance_table.to_csv(os.path.join(self.config["output_dir"], "performance_table.csv"), index=False)


class SingleLora(TrainingManager):
    """
        A subclass of TrainingManager designed for training and evaluation LoRA on a single task.
        Single-task (non continual) scenario.
    """
    def __init__(self, config: dict):
        super().__init__(config)

    def train_single_dataset(self, dataset: dict, dataset_id: int):
        """ Initialize a new LoRA and train the model with it. """
        peft_dir = os.path.join(self.config["output_dir"], str(dataset_id))
        # Pass a valid adapter name when creating a new LoRA
        # load_new_lora does not take an adapter name; it creates/attaches a fresh adapter
        self.trainer.load_new_lora()
        self.trainer.train_lora(dataset['train'], dataset['eval'], peft_dir)

        return peft_dir

    def train_single_lora(self):
        """Trains the model in a continual learning setting and tracks performance over time."""
        data_stream = prepare_task_incremental_stream(self.config["data_stream"], self.config["datasets"],
                                                      self.trainer.tokenizer.eos_token, self.config)

        task_names = self.config["datasets"]
        for dataset_id, dataset in enumerate(data_stream):
            Logger.instance().debug(f"Start training with dataset {dataset_id}: {task_names[dataset_id]}")
            peft_dir = self.train_single_dataset(dataset, dataset_id)
            Logger.instance().debug(f"End training with dataset {dataset_id}")

            # Evaluate model
            self.generator_unload_and_load_lora([peft_dir])

            current_task_performance, _ = self._evaluate_task(dataset, dataset_id)
            Logger.instance().debug(f"Current task performance: {current_task_performance}")

            # Store performance in tracking history
            self.task_performance_history[dataset_id] = current_task_performance

        performance_table = pd.DataFrame([self.task_performance_history])
        Logger.instance().debug("Task performance history:")
        Logger.instance().debug(performance_table.to_string(index=False))
        performance_table.to_csv(os.path.join(self.config["output_dir"], "performance_table.csv"), index=False)
