# callm/training_manager.py
import os
from typing import List, Optional
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from callm.data.trace.prepare_trace_data import prepare_task_incremental_stream, init_dataloader, build_mixed_stream
from callm.lora_trainer import ModelTrainer
from callm.router import Router
from callm.evaluator import Evaluator
from callm.generator import Generator
from callm.utils import Logger
from callm.prompts_repo import METRIC_SELECTION_INSTRUCTION
from unsloth import FastModel
from peft import PeftModel
import gc
import wandb
import re
from callm.utils import cast_all_lora_to, clear_unsloth_cache, move_lora_adapters_device
from callm.adapter_manager import AdapterManager
from callm.eval_pipeline import EvaluationPipeline


class TrainingManager:
    """
        Manages the training process for new or existing PEFTs.
        Decides whether to train a new PEFT or fine-tune existing ones based on the analysis of performance data and new incoming data.
        Provides methods to initiate training with selected base LLMs and PEFT techniques.
        Integrates with a training framework like Hugging Face Trainer for efficient model training.
    """

    def __init__(self, config: dict):
        self.config = config
        self.trainer = ModelTrainer(config)
        # Share base model/tokenizer with Router; Router will bypass adapters during embedding extraction
        self.router = Router(config, self.trainer.model, self.trainer.tokenizer)
        self.evaluator = Evaluator()
        self.task_performance_history = {}
        self.generator = Generator(config)
        self.metrics = {}
        
        # New managers
        self.adapter_manager = AdapterManager()
        self.eval_pipeline = EvaluationPipeline(config, self.generator, self.evaluator)
        
        # Cache for model compilation to avoid recompilation
        self._model_compiled = False

        # hard-coded "best possible" value per metric
        self.metric_scale = {
            "accuracy": 1.0,
            "rouge": 1.0,
            "sari": 100.0,
            "edim": 100.0,  # Fixed: EDIM ranges 0-100, not 0-0.8
        }

    def _unload_and_load_lora(self, peft_dir: str):
        """Switch trainer model to the specified LoRA adapter without deleting others."""
        try:
            # Prefer light-weight unload for Unsloth models; avoid deleting adapters
            if not isinstance(self.trainer.model, PeftModel) and hasattr(self.trainer.model, "unload"):
                self.trainer.model = self.trainer.model.unload()
                Logger.instance().debug("Unloaded existing adapter from trainer model (unsloth).")
        except Exception:
            Logger.instance().debug("No unloading is done because there is no active adapters on the model.")

        # Load/attach the requested LoRA adapter into the in-memory base model
        self.trainer.load_lora(peft_dir)

        # Ensure only the active adapter stays on GPU
        try:
            dev = str(next(self.trainer.model.parameters()).device)
            move_lora_adapters_device(self.trainer.model, "default", device_active=dev, device_inactive="cpu")
        except Exception:
            pass
        
        return peft_dir

    def generator_unload_and_load_lora(self, selected_peft_dirs: List, weights: Optional[List] = [1.0]):
        """Use AdapterManager to load and activate LoRA(s) on the generator model."""
        # Unload if supported to free memory (keeps adapters intact on disk)
        if self.generator.model:
            self.generator.model = self.adapter_manager.safe_unload_if_supported(self.generator.model)

        Logger.instance().debug(f"Loading adapter(s): {selected_peft_dirs}...")
        try:
            # Ensure a base model exists
            if self.generator.model is None:
                self.generator.model = self.trainer.model
                self.generator.tokenizer = self.trainer.tokenizer

            if self.config.get('adapter_merge') is None or len(selected_peft_dirs) == 1:
                # Single adapter attach
                self.generator.model = self.adapter_manager.attach_adapter(self.generator.model, selected_peft_dirs[0])
                # Keep only active adapter on GPU
                self.adapter_manager.set_active(self.generator.model, "default", device_inactive="cpu")
                self.generator.model.eval()
                Logger.instance().debug("Checkpoint adapter loaded successfully to the generator.")
            else:
                # Merge multiple adapters
                merged_name = self.adapter_manager.merge_adapters(self.generator.model, selected_peft_dirs, weights, strategy=self.config.get('adapter_merge', 'ties'))
                self.adapter_manager.set_active(self.generator.model, merged_name, device_inactive="cpu")
                self.generator.model.eval()
                Logger.instance().debug(f"Merged adapter active: {merged_name}")
        except RuntimeError as e:
            Logger.instance().critical(f"Error loading checkpoint adapter to the generator: {e}")
            exit(1)
        except FileNotFoundError as e:
            Logger.instance().critical(f"Error loading checkpoint adapter - file not found: {e}")
            exit(1)


    def _get_metric_from_prompt_llm(self, prompt: str) -> str:
        """Backwards-compatible wrapper; delegate to EvaluationPipeline."""
        try:
            return self.eval_pipeline._get_metric_from_prompt_llm(prompt)
        except Exception as e:
            Logger.instance().warning(f"Metric selection failed, defaulting to accuracy: {e}")
            return 'accuracy'

    def _evaluate_task(self, dataset: dict, dataset_id: int):
        """Evaluate using EvaluationPipeline; keep interface unchanged."""
        Logger.instance().debug(f"--- Starting Evaluation on Task: {self.config['datasets'][dataset_id]} ---")
        score, all_results = self.eval_pipeline.evaluate_task(dataset, dataset_id)
        # Sync metric name cache into TrainingManager.metrics for compatibility
        self.metrics[dataset_id] = self.eval_pipeline.metrics_cache.get(dataset_id, 'accuracy')
        Logger.instance().debug(f"--- Finished Evaluation on Task: {self.config['datasets'][dataset_id]} ---")
        return score, all_results

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

            # Use router to find the best matching LoRA based on first batch
            # Ensure router sees the latest trainer model/tokenizer (may have changed after training)
            try:
                self.router.encoder = self.trainer.model
                self.router.tokenizer = self.trainer.tokenizer
            except Exception:
                pass
            selected_peft_dirs, weights = self.router.router_strategy(prev_dataset['test'], evaluation=True)
            Logger.instance().debug(f"Selected pefts for evaluating are: {selected_peft_dirs}")
            Logger.instance().debug(f"First peft: {selected_peft_dirs[0]}")

            if not os.path.exists(selected_peft_dirs[0]):
                Logger.instance().warning(
                    f"WARNING: No LoRA found for task {prev_dataset_name}. Past performance set to 0.")
                prev_task_performance = 0.0
            else:
                # Load the selected LoRAs - single or merged
                self.generator_unload_and_load_lora(selected_peft_dirs, weights)

                # Evaluate the task with the selected LoRA
                Logger.instance().set_context(step=f"bwt-{current_dataset_id + 1}", epoch=self.config.get("num_epochs"), data=prev_dataset_name)
                prev_task_performance, _ = self._evaluate_task(prev_dataset, prev_task_id)
                Logger.instance().debug(f"  - Evaluated Performance: {prev_task_performance}")

            # Update past task performance and peft selection table
            Logger.instance().debug(
                f"  - Updating performance table at ({prev_dataset_name}, {current_dataset_id + 1}) with {prev_task_performance}")
            self.performance_table.loc[self.performance_table["Task"] == prev_dataset_name, str(
                current_dataset_id + 1)] = prev_task_performance  # Ensure col is str if necessary
            self.peft_selection_table.loc[
                self.peft_selection_table["Task"] == prev_dataset_name, str(current_dataset_id + 1)] = '-'.join(
                [adapter.split('/')[-1] for adapter in selected_peft_dirs])

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

    def train_single_dataset(self, dataset: dict):
        """Trains the model on a single dataset."""
        # Keep router encoder/tokenizer in sync with the current trainer model
        try:
            self.router.encoder = self.trainer.model
            self.router.tokenizer = self.trainer.tokenizer
        except Exception:
            pass
        selected_peft_dirs, _ = self.router.router_strategy(dataset['train'])
        Logger.instance().debug(f"Selected peft for training is: {selected_peft_dirs[0]}")
        lora_save_path = self._unload_and_load_lora(selected_peft_dirs[0])

        # Free generator references before training to avoid duplicate VRAM usage
        try:
            self.generator.model = None
            self.generator.tokenizer = None
        except Exception:
            pass
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()

        self.trainer.train_lora(dataset['train'], dataset['eval'], lora_save_path)

        return [selected_peft_dirs[0]]


    def _ensure_row_and_col(self, task_name: str, col_name: str):
        """Dynamically grow the performance / peft tables.

        Called every time we log something in the online loop.
        """
        # ––– add row if task unseen –––
        if task_name not in self.performance_table["Task"].values:
            new_row = {c: None for c in self.performance_table.columns}
            new_row["Task"] = task_name
            self.performance_table = pd.concat(
                [self.performance_table, pd.DataFrame([new_row])],
                ignore_index=True
            )
            if task_name not in ["BWT", "Average"]:
                self.peft_selection_table = pd.concat(
                    [self.peft_selection_table,
                     pd.DataFrame([{"Task": task_name}])],
                    ignore_index = True)

        # ––– add column if new step –––
        if col_name not in self.performance_table.columns:
            self.performance_table[col_name] = None
            self.peft_selection_table[col_name] = None

    def train_online(self):
        """
        Online continual-learning loop (test-then-train).

        • Evaluate on the chunk’s TRAIN set **before** updating.
        • Forgetting, BWT (= pre − best) and Moving-Avg. are logged.
        • The router is called for adapter selection only after a
          task has been seen once (same timing as original code).
        • The EXACT adapter used for pre-update evaluation is the one
          that is fine-tuned on this chunk (Option A).
        • No post-update evaluation is performed.
        """

        # ─────────────────── build chunk stream ─────────────────────
        chunk_stream = build_mixed_stream(
            data_stream=self.config["data_stream"],
            dataset_names=self.config["datasets"],
            eos_token=self.trainer.tokenizer.eos_token,
            config=self.config,
            chunk_size=self.config["chunk_size"],
            schedule="random",
            batch_size=self.config["batch_size"],
            max_chunks=self.config["max_chunks"],
            seed=int(self.config.get("stream_seed", 42)),
        )

        # ───────────────────── tracking tables ──────────────────────
        self.performance_table = pd.DataFrame(columns=["Task", "Metric_GT", "Metric"])
        for helper in ["Forgetting", "BWT", "Moving Avg."]:
            self.performance_table.loc[len(self.performance_table)] = [helper, "", ""]
        self.peft_selection_table = pd.DataFrame(columns=["Task"])

        self.task_performance_history = {}  # best score per task
        self.running_sum = self.running_cnt = 0.0  # for Moving-Avg.

        # ───── helpers ─────
        def _ensure_col(c):
            if c not in self.performance_table.columns:
                self.performance_table[c] = np.nan
                self.peft_selection_table[c] = np.nan

        def _row(task):  # only “pre” row kept
            return f"{task} (pre)"

        def _add_task_rows(task):
            if not (self.performance_table["Task"] == _row(task)).any():
                blank = {col: np.nan for col in self.performance_table.columns}
                blank["Task"] = _row(task)
                self.performance_table.loc[len(self.performance_table)] = blank
            if not (self.peft_selection_table["Task"] == task).any():
                self.peft_selection_table.loc[len(self.peft_selection_table)] = {"Task": task}

        # ───────────────────────── main loop ─────────────────────────
        for step, (task, chk, tr, ev, te) in enumerate(chunk_stream, start=1):
            col = str(step)
            _ensure_col(col)
            _add_task_rows(task)

            task_idx = list(self.config["datasets"]).index(task)
            prev_best = self.task_performance_history.get(task)  # None on first sighting

            # ── 0. make sure generator has a usable base model ───────────
            if self.generator.model is None or self.generator.tokenizer is None:
                self.generator.model = self.trainer.model
                self.generator.tokenizer = self.trainer.tokenizer
                self.generator.model = FastModel.for_inference(self.generator.model)

            # ── 1. choose adapter(s) ONLY if task seen before ─────────────
            sel_pefts, sel_wts = [], []
            if prev_best is not None:
                # Router should use the freshest trainer model (now a PeftModel with adapters)
                try:
                    self.router.encoder = self.trainer.model
                    self.router.tokenizer = self.trainer.tokenizer
                except Exception:
                    pass
                sel_pefts, sel_wts = self.router.router_strategy(tr, evaluation=True)
                sel_pefts = [p for p in sel_pefts if p and os.path.isdir(p)]
                if sel_pefts:
                    self.generator_unload_and_load_lora(sel_pefts, sel_wts)
            # first sighting → stay with current (base) model


            # ── 2. evaluate on TRAIN set (pre-update) ────────────────────
            # Set detail-log context so per-example lines are prefixed
            Logger.instance().set_context(step=step, epoch="pre", data=task)
            pre_perf, _ = self._evaluate_task({"test": tr}, task_idx)
            self.performance_table.loc[self.performance_table["Task"] == _row(task), col] = pre_perf

            # write metric name once (GT metric vs selected/auto metric)
            if self.performance_table.loc[self.performance_table["Task"] == _row(task), "Metric"].isna().all():
                gt_metric = None
                if len(self.config.get("evaluation_metrics", [])) > task_idx:
                    gt_metric = self.config["evaluation_metrics"][task_idx]
                self.performance_table.loc[self.performance_table["Task"] == _row(task), ["Metric_GT"]] = gt_metric
                self.performance_table.loc[self.performance_table["Task"] == _row(task), ["Metric"]] = self.metrics.get(task_idx, gt_metric or "accuracy")

            # ── 3. helper rows ───────────────────────────────────────────
            if prev_best is not None:
                metric_name = self.metrics[task_idx]
                scale = self.metric_scale.get(metric_name, 1.0)

                forgetting = (prev_best - pre_perf) / scale
                bwt = (pre_perf - prev_best) / scale

                self.performance_table.loc[self.performance_table["Task"] == "Forgetting", col] = round(forgetting, 3)
                self.performance_table.loc[self.performance_table["Task"] == "BWT", col] = round(bwt, 3)

            # moving average
            metric_name = self.metrics[task_idx]
            norm_score = pre_perf / self.metric_scale.get(metric_name, 1.0)
            self.running_sum += norm_score
            self.running_cnt += 1
            self.performance_table.loc[self.performance_table["Task"] == "Moving Avg.", col] = \
                round(self.running_sum / self.running_cnt, 3)

            # ── 4. update best-so-far record ─────────────────────────────
            self.task_performance_history[task] = max(prev_best or -float("inf"), pre_perf)

            # ── 5. fine-tune **the same adapter** we just evaluated ─────
            #     (Option A – no API change to train_single_dataset)
            if sel_pefts:  # existing adapter used in evaluation
                lora_path = self._unload_and_load_lora(sel_pefts[0])
            else:  # first chunk → create a new adapter dir
                # router in "training" mode (can create prototype)
                try:
                    self.router.encoder = self.trainer.model
                    self.router.tokenizer = self.trainer.tokenizer
                except Exception:
                    pass
                new_peft_dirs, _ = self.router.router_strategy(tr)  # evaluation=False
                lora_path = self._unload_and_load_lora(new_peft_dirs[0])
                sel_pefts = [new_peft_dirs[0]]  # remember for logging

            # Ensure model is in training mode before training
            # Since generator.model and trainer.model are the same object, and we call
            # generator.model.eval() after each training iteration, the model stays in eval mode
            # for subsequent training iterations, which disables dropout and affects training dynamics
            self.trainer.model.train()
            Logger.instance().debug(f"Model set to training mode before training iteration {step}")
            
            # actually train
            self.trainer.train_lora(tr, ev, lora_path)

            # Use the freshly-trained in-memory model directly for generation to avoid reloading + recompilation
            self.generator.model = self.trainer.model
            self.generator.tokenizer = self.trainer.tokenizer
            
            # Only compile for inference if not already compiled
            if not self._model_compiled:
                try:
                    self.generator.model = FastModel.for_inference(self.generator.model)
                    self._model_compiled = True
                    Logger.instance().debug("Model compiled for inference")
                except Exception as e:
                    Logger.instance().debug(f"Model compilation failed: {e}")
            
            self.generator.model.eval()

            self.peft_selection_table.loc[self.peft_selection_table["Task"] == task, col] = \
                os.path.basename(sel_pefts[0])

            # ─────────────────────────────────────────────────────
            #  MEMORY CLEANUP (unified)
            # ─────────────────────────────────────────────────────
            from callm.memory import MemoryManager
            stats = MemoryManager.cleanup(label="training.loop")
            if stats is not None:
                Logger.instance().debug(
                    f"GPU memory after cleanup: allocated {stats.allocated_gb:.2f} GB | reserved {stats.reserved_gb:.2f} GB"
                )
            # ─────────────────────────────────────────────────────

            # ── 6. console snapshot (optional) ───────────────────────────
            Logger.instance().debug("\nPerformance table:\n" +
                                    self.performance_table.replace(np.nan, "-").set_index("Task").to_string() + "\n")
            Logger.instance().debug("\nPEFT selection table:\n" +
                                    self.peft_selection_table.set_index("Task").to_string() + "\n")


        # ───────────────────────── save CSVs ───────────────────────────
        # Log tables in W&B only when explicitly enabled.
        if self.config.get("wandb_name"):
            if wandb.run is None:
                wandb.init(project=self.config["wandb_name"], entity=self.config.get("wandb_project"))
            performance_table = wandb.Table(dataframe=self.performance_table)
            wandb.log({"performance table": performance_table})
            peft_table = wandb.Table(dataframe=self.peft_selection_table)
            wandb.log({"peft selection table": peft_table})

        # Log tables locally
        out_dir = self.config["output_dir"]
        self.performance_table.to_csv(os.path.join(out_dir, "performance_table_online.csv"), index=False)
        self.peft_selection_table.to_csv(os.path.join(out_dir, "peft_selection_table_online.csv"), index=False)

    def train_batched(self):
        """
        Your previous batched-incremental routine, lifted verbatim out of the
        old train_continual so that train_continual can now simply dispatch.
        """
        # ------------------------------------------------------------------
        # 1) Build one-task-at-a-time stream
        # ------------------------------------------------------------------
        data_stream = prepare_task_incremental_stream(
            self.config["data_stream"],
            self.config["datasets"],
            self.trainer.tokenizer.eos_token,
            self.config,
        )

        # ------------------------------------------------------------------
        # 2) Initialise tracking tables exactly like before
        # ------------------------------------------------------------------
        task_names = self.config["datasets"]
        self.performance_table = pd.DataFrame(
            columns=["Task", "Metric_GT", "Metric"]
                    + [str(i) for i in range(1, len(data_stream) + 1)]
        )
        self.performance_table["Task"] = task_names
        self.performance_table["Metric_GT"] = self.config.get(
            "evaluation_metrics", [None] * len(task_names)
        )
        self.performance_table["Metric"] = None

        # special rows
        self.performance_table.loc[len(self.performance_table)] = ["BWT", None, None] \
                                                                    + [None] * len(data_stream)
        self.performance_table.loc[len(self.performance_table)] = ["Average", None, None] \
                                                                    + [None] * len(data_stream)

        # PEFT-selection log
        self.peft_selection_table = pd.DataFrame(
            columns=["Task"] + [str(i) for i in range(1, len(data_stream) + 1)]
        )
        self.peft_selection_table["Task"] = task_names

        # ------------------------------------------------------------------
        # 3) Main batched loop (same as before)
        # ------------------------------------------------------------------
        for dataset_id, dataset in enumerate(data_stream):
            Logger.instance().debug(
                f"Start training with dataset {dataset_id}: {task_names[dataset_id]}"
            )

            # Keep router synced before selection/training
            try:
                self.router.encoder = self.trainer.model
                self.router.tokenizer = self.trainer.tokenizer
            except Exception:
                pass

            selected_peft = self.train_single_dataset(dataset)

            # evaluate *after* training on the current task
            self.generator_unload_and_load_lora(selected_peft)
            # Set detail-log context for post-update eval
            Logger.instance().set_context(step=dataset_id + 1, epoch=self.config.get("num_epochs"), data=task_names[dataset_id])
            current_task_performance, _ = self._evaluate_task(dataset, dataset_id)
            self.task_performance_history[dataset_id] = current_task_performance

            # update tables
            self.performance_table.loc[
                self.performance_table["Task"] == task_names[dataset_id], str(dataset_id + 1)
            ] = current_task_performance

            self.performance_table.loc[
                self.performance_table["Task"] == task_names[dataset_id], "Metric"
            ] = self.metrics[dataset_id]

            self.peft_selection_table.loc[
                self.peft_selection_table["Task"] == task_names[dataset_id], str(dataset_id + 1)
            ] = selected_peft[0].split("/")[-1]

            # backward transfer on all *previous* tasks
            bwt_score, _ = self._calculate_backward_transfer(data_stream, dataset_id)
            self.performance_table.loc[
                self.performance_table["Task"] == "BWT", str(dataset_id + 1)
            ] = bwt_score

            # ---------------------------------------------------------------
            # Re-compute “Average” column on a 0-1 scale
            # ---------------------------------------------------------------
            numeric_cols = self.performance_table.columns.difference(
                ["Task", "Metric_GT", "Metric"]
            )

            task_rows = self.performance_table[
                ~self.performance_table["Task"].isin(["BWT", "Average", "Pre-update"])
            ]

            # Build a normalised copy without touching the original table
            def _norm_row(row):
                scale = self.metric_scale.get(row["Metric"], 1.0)  # fallback = 1.0
                return row[numeric_cols].astype(float) / scale

            norm_df = task_rows.apply(_norm_row, axis=1)
            avg_norm = norm_df.mean(skipna=True)

            self.performance_table.loc[
                self.performance_table["Task"] == "Average", numeric_cols
            ] = avg_norm.round(2)  # keep two decimals

            Logger.instance().debug("\nUpdated performance table:\n"
                                        f"{self.performance_table.to_string(index=False)}\n")
            Logger.instance().debug("\nPEFT selection table:\n"
                                        f"{self.peft_selection_table.to_string(index=False)}\n")

        # ───────────────────────── save CSVs ───────────────────────────
        # Log tables in W&B only when explicitly enabled.
        if self.config.get("wandb_name"):
            if wandb.run is None:
                wandb.init(project=self.config["wandb_name"], entity=self.config.get("wandb_project"))
            performance_table = wandb.Table(dataframe=self.performance_table)
            wandb.log({"performance table batched": performance_table})
            peft_table = wandb.Table(dataframe=self.peft_selection_table)
            wandb.log({"peft selection table batched": peft_table})

        self.performance_table.to_csv(
            os.path.join(self.config["output_dir"], "performance_table_batched.csv"),
            index=False,
        )
        self.peft_selection_table.to_csv(
            os.path.join(self.config["output_dir"], "peft_selection_table_batched.csv"),
            index=False,
        )
        
        # Return performance table for result extraction
        return self.performance_table

    def train_continual(self):
        mode = self.config["stream_type"].lower()
        if mode == "batched":
            return self.train_batched()
        elif mode == "online":
            return self.train_online()
        else:
            raise ValueError(f"Unknown stream_type '{self.config['stream_type']}'")
