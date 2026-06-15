from datasets import load_dataset
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from typing import Dict, List, Tuple, Iterator
import numpy as np
from datasets import Dataset, concatenate_datasets

DEVICE = "cuda"

from callm.utils import Logger
from callm.prompts_repo import PromptBuilder


TRACE_METRICS = {
    "C-STANCE": "accuracy",
    "FOMC": "accuracy",
    "MeetingBank": "rouge",
    "Py150": "edim",
    "ScienceQA": "accuracy",
    "NumGLUE-cm": "accuracy",
    "NumGLUE-ds": "accuracy",
    "20Minuten": "sari"
}

TRACE_METRICS_DEBUG = {
    "C-STANCE": "accuracy",
    "FOMC": "accuracy",
    "MeetingBank": "rouge"
}


def get_trace_metric_for_dataset(dataset_name: str, debug: bool = False) -> str:
    """Get the ground truth metric for a TRACE dataset.
    
    Args:
        dataset_name: The TRACE dataset name (e.g., 'C-STANCE', 'MeetingBank')
        debug: Whether to use debug mappings
        
    Returns:
        The metric name for the dataset, or None if not found
    """
    metrics_map = TRACE_METRICS_DEBUG if debug else TRACE_METRICS
    return metrics_map.get(dataset_name)

def validate_metrics_trace(config):
    dataset_metric_map = TRACE_METRICS_DEBUG if config["debug"] else TRACE_METRICS

    datasets = list(config["datasets"])
    metrics = list(config["evaluation_metrics"])

    if len(datasets) != len(metrics):
        Logger.instance().error("Datasets and metrics do not match!")
        raise ValueError("Datasets and metrics do not match!")

    for dataset, metric in zip(datasets, metrics):
        if dataset not in dataset_metric_map or dataset_metric_map[dataset] != metric:
            Logger.instance().error("Datasets and metrics do not match!")
            raise ValueError("Datasets and metrics do not match!")


def prepare_single_dataset_prompt_format(data, eos_token: str, debug: bool, prompt_style: str = "alpaca"):
    """Format a dataset split using the centralized PromptBuilder (same template as inference)."""
    eos = eos_token or ""

    def formatting_prompts_func(examples):
        texts = []
        for instruction, output in zip(examples["prompt"], examples["answer"]):
            text = PromptBuilder.build_generation_prompt(
                instruction=str(instruction),
                style=prompt_style,
                response=str(output),
            ) + eos
            texts.append(text)
        return {"text": texts}

    return data.map(formatting_prompts_func, batched=True)


def prepare_single_dataset_alpaca_format(data: str, eos_token: str, debug: bool, prompt_style: str = "alpaca"):
    data_files = {
        'train': f'{data}/train.json',
        'eval': f'{data}/eval.json',
        'test': f'{data}/test.json',
    }

    dataset_name = data.split('/')[-1]
    Logger.instance().debug(f"[Dataset] Loading {dataset_name} dataset files...")

    import os
    for split, file_path in data_files.items():
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            Logger.instance().debug(f"[Dataset] {dataset_name} {split}.json: {size_mb:.1f} MB")
        else:
            Logger.instance().warning(f"[Dataset] {dataset_name} {split}.json: FILE NOT FOUND")

    try:
        ds = load_dataset('json', data_files=data_files)
        Logger.instance().debug(f"[Dataset] Successfully loaded {dataset_name} dataset")
    except Exception as e:
        Logger.instance().error(f"[Dataset] Failed to load {dataset_name}: {e}")
        raise

    ds['train'] = prepare_single_dataset_prompt_format(ds['train'], eos_token, debug, prompt_style)
    ds['test'] = prepare_single_dataset_prompt_format(ds['test'], eos_token, debug, prompt_style)
    ds['eval'] = prepare_single_dataset_prompt_format(ds['eval'], eos_token, debug, prompt_style)

    return ds


def prepare_task_incremental_stream(data_stream: str, datasets: List[str], eos_token: str, config: dict):
    # Verify that the datasets and the metrics corresponds
    #validate_metrics_trace(config)

    if "TRACE" in data_stream:
        # Handle TRACE datasets (existing logic)
        stream = []
        for dataset in datasets:
            if data_stream == "TRACE-500":
                Logger.instance().debug("Training with TRACE-500")
                dataset_path = f"callm/data/trace/LLM-CL-Benchmark_500/{dataset}"
            elif data_stream == "TRACE-1000":
                Logger.instance().debug("Training with TRACE-1000")
                dataset_path = f"callm/data/trace/LLM-CL-Benchmark_1000/{dataset}"
            elif data_stream == "TRACE-5000":
                Logger.instance().debug("Training with TRACE-5000")
                dataset_path = f"callm/data/trace/LLM-CL-Benchmark_5000/{dataset}"
            else:
                raise ValueError(f"The TRACE data stream '{data_stream}' is not currently supported.")

            prompt_style = config.get("prompt_style", "alpaca")
            stream.append(prepare_single_dataset_alpaca_format(dataset_path, eos_token, config["debug"], prompt_style))
        return stream

    elif "CITB" in data_stream:
        # Handle CITB datasets
        from callm.data.citb.prepare_ni_data import prepare_stream_citb
        config_path = "callm/data/citb/ni_data_config.json"
        prompt_style = config.get("prompt_style", "alpaca")
        stream, task_names = prepare_stream_citb(data_stream, config_path, eos_token, config.get("synthetic_data_path"), config["debug"], prompt_style)
        return stream
        
    else:
        raise ValueError(f"The data stream '{data_stream}' is not currently supported. You can create a custom stream and add it to the list.")


def init_dataloader(dataset: dict, batch_size: int):
    def _loader(split):
        if split not in dataset:
            return None
        ds = dataset[split].with_format("torch", device=DEVICE)
        sampler = RandomSampler(ds) if split == "train" else SequentialSampler(ds)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)

    return _loader("train"), _loader("eval"), _loader("test")


#online data


def _pad(idx: np.ndarray, target_len: int, rng: np.random.Generator):
    """Pad idx to exactly target_len by sampling with replacement."""
    if len(idx) >= target_len:
        return idx
    extra = rng.choice(idx, target_len - len(idx), replace=True)
    return np.concatenate([idx, extra])


def split_per_task(
    ds_dicts: Dict[str, Dict[str, Dataset]],
    chunk_size: int,                      # must be divisible by 4
    *,
    seed: int = 42,
) -> List[Tuple[str, int, Dict[str, Dataset]]]:
    """
    Build timeline for an online continual-learning stream.

    Guarantees per (task, chunk):

        • Full chunks:                train = chunk_size
                                      eval  = chunk_size/4
                                      test  = chunk_size/4
        • Last chunk (if any):        receives *all* remaining rows.
    Never discards data.  Never emits a chunk with an empty train split.
    """
    assert chunk_size % 4 == 0, "`chunk_size` must be divisible by 4"
    dev_chunk = chunk_size // 4

    rng = np.random.default_rng(seed)
    timeline: List[Tuple[str, int, Dict[str, Dataset]]] = []

    for task, splits in ds_dicts.items():
        n_train, n_eval, n_test = map(len, splits.values())
        if min(n_train, n_eval, n_test) == 0:
            raise ValueError(f"{task}: one of the splits is empty.")

        # ------------------------------------------------------------------ #
        # 1) Shuffle indices once per split
        # ------------------------------------------------------------------ #
        shuffled = {
            name: rng.permutation(len(ds))
            for name, ds in splits.items()
        }

        # ------------------------------------------------------------------ #
        # 2) Determine number of *full* chunks driven by train size
        # ------------------------------------------------------------------ #
        n_full = n_train // chunk_size
        rem_train = n_train - n_full * chunk_size     # 0 … chunk_size-1

        # Ensure enough dev data for full chunks by padding smaller dev sets
        needed_dev = n_full * dev_chunk
        shuffled["eval"] = _pad(shuffled["eval"], needed_dev, rng)
        shuffled["test"] = _pad(shuffled["test"], needed_dev, rng)

        # Slice out the dev rows for full chunks
        full_eval = shuffled["eval"][:needed_dev]
        full_test = shuffled["test"][:needed_dev]
        rest_eval = shuffled["eval"][needed_dev:]     # leftovers
        rest_test = shuffled["test"][needed_dev:]

        # ------------------------------------------------------------------ #
        # 3) Emit the FULL chunks (exact 4 : 1 : 1)
        # ------------------------------------------------------------------ #
        for c in range(n_full):
            t_slice = shuffled["train"][c * chunk_size : (c + 1) * chunk_size]
            e_slice = full_eval[c * dev_chunk : (c + 1) * dev_chunk]
            te_slice = full_test[c * dev_chunk : (c + 1) * dev_chunk]

            chunk_dict = {
                "train": splits["train"].select(t_slice),
                "eval":  splits["eval"].select(e_slice),
                "test":  splits["test"].select(te_slice),
            }
            timeline.append((task, c, chunk_dict))

        # ------------------------------------------------------------------ #
        # 4) Handle leftovers
        # ------------------------------------------------------------------ #
        if rem_train:                      # → real final chunk with train rows
            cid = n_full
            chunk_dict = {
                "train": splits["train"].select(shuffled["train"][-rem_train:]),
                "eval":  splits["eval"].select(rest_eval),
                "test":  splits["test"].select(rest_test),
            }
            timeline.append((task, cid, chunk_dict))

        elif len(rest_eval) or len(rest_test):
            # Only dev leftovers.  Merge them into *last* full chunk.
            last_task, last_cid, last_dict = timeline[-1]
            assert last_task == task       # safety
            if len(rest_eval):
                last_dict["eval"] = concatenate_datasets(
                    [last_dict["eval"], splits["eval"].select(rest_eval)]
                )
            if len(rest_test):
                last_dict["test"] = concatenate_datasets(
                    [last_dict["test"], splits["test"].select(rest_test)]
                )

    return timeline

def interleave(
    timeline: List[Tuple[str, int, Dict[str, Dataset]]],
    strategy: str = "random",
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    if strategy == "random":
        rng.shuffle(timeline)

    elif strategy == "round_robin":
        # bucket by task then take one per round
        buckets = {}
        for entry in timeline:
            buckets.setdefault(entry[0], []).append(entry)
        shuffled = []
        while buckets:
            for t in list(buckets):
                shuffled.append(buckets[t].pop(0))
                if not buckets[t]:
                    buckets.pop(t)
        timeline = shuffled

    else:
        raise ValueError(f"unknown strategy {strategy}")

    return timeline

def chunk_stream(
    interleaved: List[Tuple[str, int, Dict[str, Dataset]]],
    batch_size: int,
) -> Iterator[Tuple[str, int, DataLoader, DataLoader, DataLoader]]:
    """
    Yields (task_id, local_chunk_id,
            train_loader, eval_loader, test_loader)
    `train_loader` uses RandomSampler; eval/test use SequentialSampler.
    """
    for task, cid, splits in interleaved:
        yield task, cid, splits["train"], splits["eval"], splits["test"]

def renumber_by_occurrence(
    interleaved: List[Tuple[str, int, dict]]
) -> List[Tuple[str, int, dict]]:
    counters = defaultdict(int)          # task → next local id
    new_timeline = []

    for task, _old_cid, chunk_dict in interleaved:
        new_cid = counters[task]         # how many times we've seen this task
        counters[task] += 1
        new_timeline.append((task, new_cid, chunk_dict))

    return new_timeline

def build_mixed_stream(
    data_stream: str,
    dataset_names: List[str],
    eos_token: str,
    config: dict,
    *,
    chunk_size: int = 128,
    schedule: str = "random",
    batch_size: int = 8,
    max_chunks: int=50,
    seed: int = 42,
):
    # ---------- load & format every split, add task_id column ----------
    per_task_splits = {}
    
    if "TRACE" in data_stream:
        # Handle TRACE datasets (existing logic)
        prompt_style = config.get("prompt_style", "alpaca")
        for name in dataset_names:
            root = f"callm/data/trace/LLM-CL-Benchmark_{data_stream.split('-')[1]}/{name}"
            ds = prepare_single_dataset_alpaca_format(root, eos_token, config["debug"], prompt_style)
            for split in ds:                       # add task column everywhere
                ds[split] = ds[split].add_column("task_id", [name] * len(ds[split]))
            per_task_splits[name] = ds             # {"train","eval","test"}

    elif "CITB" in data_stream:
        # Handle CITB datasets
        from callm.data.citb.prepare_ni_data import prepare_stream_citb
        config_path = "callm/data/citb/ni_data_config.json"
        prompt_style = config.get("prompt_style", "alpaca")
        stream, task_names = prepare_stream_citb(
            data_stream,
            config_path,
            eos_token,
            config.get("synthetic_data_path"),
            config["debug"],
            prompt_style,
        )

        # Deterministic RNG for any sampling/padding we do during stream construction
        rng = np.random.default_rng(seed)

        for name, ds in zip(dataset_names, stream):
            # CITB datasets now have proper train/eval/test splits from prepare_stream_citb

            # --- CITB online chunk diagnostics + stabilization ---
            # With synthetic v0.6, some tasks end up with train sizes like 3501, 3499, 3107, etc.
            # When chunk_size=100 this produces a final remainder chunk (train=1/7/99/...) which can
            # randomly appear early due to interleaving and crash trainers.
            #
            # To stabilize CITB online runs, we make train length divisible by chunk_size by choosing
            # the smaller adjustment between:
            #   - dropping `rem` examples
            #   - padding `chunk_size - rem` examples (sampling with replacement)
            train_len = len(ds["train"])
            rem = int(train_len) % int(chunk_size)
            if rem != 0:
                drop_n = rem
                pad_n = int(chunk_size) - rem

                # If train_len < chunk_size, dropping would make it empty; always pad to one full chunk.
                if train_len < int(chunk_size):
                    base_idx = np.arange(train_len)
                    extra_idx = rng.choice(base_idx, size=pad_n, replace=True)
                    new_idx = np.concatenate([base_idx, extra_idx])
                    ds["train"] = ds["train"].select(list(new_idx))
                    Logger.instance().warning(
                        f"[CITB][OnlineStream] {name}: train={train_len} < chunk_size={chunk_size}; "
                        f"padded +{pad_n} -> {len(ds['train'])}"
                    )
                else:
                    keep_n = train_len - drop_n

                    # Prefer the smaller perturbation, but never drop to 0.
                    if pad_n < drop_n or keep_n <= 0:
                        base_idx = np.arange(train_len)
                        extra_idx = rng.choice(base_idx, size=pad_n, replace=True)
                        new_idx = np.concatenate([base_idx, extra_idx])
                        ds["train"] = ds["train"].select(list(new_idx))
                        Logger.instance().warning(
                            f"[CITB][OnlineStream] {name}: train={train_len} not divisible by chunk_size={chunk_size}; "
                            f"padded +{pad_n} -> {len(ds['train'])}"
                        )
                    else:
                        keep_idx = rng.permutation(train_len)[:keep_n]
                        ds["train"] = ds["train"].select(list(keep_idx))
                        Logger.instance().warning(
                            f"[CITB][OnlineStream] {name}: train={train_len} not divisible by chunk_size={chunk_size}; "
                            f"dropped -{drop_n} -> {len(ds['train'])}"
                        )

            Logger.instance().debug(
                f"[CITB][OnlineStream] {name}: train={len(ds['train'])} eval={len(ds['eval'])} test={len(ds['test'])} "
                f"(chunk_size={chunk_size}, dev_chunk={int(chunk_size)//4})"
            )

            # Add task_id column to all splits
            for split in ds:
                ds[split] = ds[split].add_column("task_id", [name] * len(ds[split]))
            per_task_splits[name] = ds
    
    else:
        raise ValueError(f"The data stream '{data_stream}' is not currently supported.")

    # ---------- aligned slicing & interleaving ----------
    timeline  = split_per_task(per_task_splits, chunk_size, seed=seed)
    mixed     = interleave(timeline, schedule, seed)  # task order decided here with seed
    mixed     = renumber_by_occurrence(mixed)         # ← NEW LINE

    # Limit the number of chunks
    if max_chunks is not None and max_chunks > 0:
        mixed = mixed[:max_chunks]

    # ---------- online iterator ----------
    return chunk_stream(mixed, batch_size)
