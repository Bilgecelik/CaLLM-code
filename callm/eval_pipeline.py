from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import re
from callm.evaluator import Evaluator
from callm.generator import Generator
from callm.prompts_repo import METRIC_SELECTION_INSTRUCTION
from callm.utils import Logger


@dataclass
class EvalConfig:
    use_gt_metrics: bool
    evaluation_metrics: List[str]
    gen_batch_size: int = 16
    gen_max_new_tokens_default: int = 128


class EvaluationPipeline:
    """
    Centralizes metric selection (GT or LLM-based) and evaluation using
    the Generator and Evaluator. Also handles prompt formatting policy
    (e.g., whether to prepend MC instruction based on the metric).
    """
    def __init__(self, config: dict, generator: Generator, evaluator: Evaluator):
        self.config = config
        self.generator = generator
        self.evaluator = evaluator
        self.metrics_cache: Dict[int, str] = {}

    def _get_metric_from_prompt_llm(self, prompt: str) -> str:
        """Select metric via LLM using a raw meta-prompt (no MC instruction)."""
        max_chars = 700
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars] + " …"

        llm_instruction = METRIC_SELECTION_INSTRUCTION.format(prompt=prompt)
        # Use raw generation without prompt wrapping
        raw = self.generator.generate_raw([llm_instruction])[0]

        # Parse robustly
        parts = re.split(r"###\s*response\s*:\s*", raw, flags=re.IGNORECASE, maxsplit=1)
        response = (parts[1] if len(parts) > 1 else raw).lower()

        Logger.instance().debug(f"[MetricSel] Full LLM response: {raw}")
        Logger.instance().debug(f"[MetricSel] Parsed: {response}")

        for key in ["sari", "accuracy", "rouge", "edim"]:
            if key in response:
                return key

        Logger.instance().warning("[MetricSel] No metric detected. Falling back to 'accuracy'.")
        return "accuracy"

    def _get_gt_metric_from_mappings(self, dataset_id: int) -> str:
        """Get ground truth metric using centralized data stream mappings."""
        data_stream = self.config.get("data_stream")
        if not data_stream:
            return None
            
        try:
            # Try TRACE datasets first
            if "TRACE" in data_stream:
                from callm.data.trace.prepare_trace_data import get_trace_metric_for_dataset
                dataset_name = self.config.get("datasets", [None])[dataset_id] if dataset_id < len(self.config.get("datasets", [])) else None
                if dataset_name:
                    return get_trace_metric_for_dataset(dataset_name, self.config.get("debug", False))
            
            # Try CITB datasets
            elif "CITB" in data_stream:
                from callm.data.citb.prepare_ni_data import get_citb_metric_for_stream
                return get_citb_metric_for_stream(data_stream)
                
        except Exception as e:
            Logger.instance().debug(f"[MetricSel] Error getting GT metric from mappings: {e}")
            
        return None

    def get_metric_for_dataset(self, dataset: dict, dataset_id: int) -> str:
        if dataset_id in self.metrics_cache:
            return self.metrics_cache[dataset_id]

        # GT metrics if requested and available
        if self.config.get("use_gt_metrics", False):
            # First try centralized mappings
            gt_metric = self._get_gt_metric_from_mappings(dataset_id)
            if gt_metric:
                Logger.instance().debug(f"[MetricSel] Using GT metric '{gt_metric}' from data stream mapping")
                self.metrics_cache[dataset_id] = gt_metric
                return gt_metric
                
            # Fallback to evaluation_metrics config if available
            if len(self.config.get("evaluation_metrics", [])) > dataset_id:
                metric = self.config["evaluation_metrics"][dataset_id]
                Logger.instance().debug(f"[MetricSel] Using GT metric '{metric}' from evaluation_metrics config")
                self.metrics_cache[dataset_id] = metric
                return metric
                
            # No GT mapping found, warn and fall back to auto detection
            data_stream = self.config.get("data_stream", "unknown")
            Logger.instance().warning(f"[MetricSel] No GT metric mapping found for data stream '{data_stream}'. Falling back to auto-detection.")

        # Auto metric selection: sample a few prompts from test split
        sample_cnt = min(3, len(dataset["test"]))
        for i in range(sample_cnt):
            try:
                metric = self._get_metric_from_prompt_llm(dataset["test"][i]["prompt"])  # type: ignore
                self.metrics_cache[dataset_id] = metric
                return metric
            except Exception as e:
                Logger.instance().debug(f"[MetricSel] trial {i+1}/{sample_cnt} failed: {e}")

        # Fallback
        self.metrics_cache[dataset_id] = "accuracy"
        return "accuracy"

    def evaluate_task(self, dataset: dict, dataset_id: int) -> Tuple[float, List[dict]]:
        """
        Runs generation over the test split, extracts answers, and computes the
        chosen metric. Applies MC instruction only when metric == 'accuracy'.
        """
        from callm.memory import MemoryManager
        
        # Cleanup memory before starting evaluation
        MemoryManager.cleanup(label="evaluation.start")
        
        # Set detail-log context with dataset name; step/epoch may be set by caller
        try:
            ds_name = self.config.get("datasets", [None])[dataset_id]
            if ds_name is not None:
                Logger.instance().set_context(data=ds_name)
        except Exception:
            pass

        metric = self.get_metric_for_dataset(dataset, dataset_id)

        # Determine generation length:
        # - If user set gen_max_new_tokens, use it.
        # - Otherwise, default to the generator's default (256) instead of metric-based mapping.
        override = self.config.get("gen_max_new_tokens", None)
        if override is not None:
            gen_max_new = int(override)
        else:
            gen_max_new = 256
        if gen_max_new <= 0:
            gen_max_new = 1
        Logger.instance().debug(f"[Eval] gen_max_new_tokens={gen_max_new} (metric={metric}, override={override})")

        # Fixed generation batch size (no dynamic adjustment)
        batch_size = int(self.config.get("gen_batch_size", 16))

        all_results: List[dict] = []
        # Iterate over dataset['test'] in batches
        n = len(dataset["test"])  # type: ignore
        total_batches = (n + batch_size - 1) // batch_size
        Logger.instance().debug(f"[Eval] Processing {n} test samples in {total_batches} batches of size {batch_size}")
        
        logged_milestones = set()  # Track which 10% milestones we've already logged
        
        for i in range(0, n, batch_size):
            batch_num = i // batch_size + 1
            progress_pct = int((batch_num / total_batches) * 100)
            
            # Calculate the 10% milestone (10, 20, 30, etc.)
            milestone = (progress_pct // 10) * 10
            
            # Log only at 10% milestones (once each) plus first and last batch
            should_log = (
                (milestone >= 10 and milestone % 10 == 0 and milestone not in logged_milestones) or
                batch_num == 1 or 
                batch_num == total_batches
            )
            
            if should_log:
                if milestone >= 10 and milestone % 10 == 0:
                    logged_milestones.add(milestone)
                samples_range = f"{i}-{min(i + batch_size - 1, n-1)}"
                Logger.instance().debug(f"[Eval] Batch {batch_num}/{total_batches} ({milestone}%) - samples {samples_range}")
            
            test_split = dataset["test"]  # type: ignore
            end = min(i + batch_size, n)

            # Avoid per-batch Dataset.select() overhead where possible.
            try:
                batch_dict = test_split[i:end]  # HF Dataset slicing returns dict-of-lists
                prompts = list(batch_dict["prompt"])
                answers = list(batch_dict["answer"])
            except Exception:
                try:
                    batch = test_split.select(range(i, end))  # type: ignore
                    prompts = [ex["prompt"] for ex in batch]
                    answers = [ex["answer"] for ex in batch]
                except Exception as e:
                    Logger.instance().error(f"[Eval] Failed to process batch {batch_num}: {e}")
                    raise

            generated_texts = self.generator.generate_answers(
                prompts,
                max_new_tokens=gen_max_new,
                use_cache=True,
            )

            for prompt, gt, gen_text in zip(prompts, answers, generated_texts):
                extracted = self.generator.extract_answer(gen_text)
                all_results.append({
                    "input": prompt,
                    "model_answer": extracted,
                    "ground_truth": gt,
                })

        # Compute metric
        evaluator_map = {
            "sari": self.evaluator.calculate_sari,
            "accuracy": self.evaluator.calculate_accuracy,
            "rouge": self.evaluator.calculate_rouge,
            "edim": self.evaluator.calculate_edim,
        }
        score = evaluator_map[metric](all_results)
        
        # Cleanup memory after evaluation
        MemoryManager.cleanup(label="evaluation.end")
        
        return score, all_results
