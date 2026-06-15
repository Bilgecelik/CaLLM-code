import os
import sys
import logging
import torch
from torch import nn
import torch.nn.functional as F
import shutil
from collections import Counter
import types
from peft.tuners.lora.layer import LoraLayer

# Global variables
def _load_hf_token():
    """Read an optional Hugging Face token without requiring a tracked file."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token.strip()

    token_path = os.path.join(os.getcwd(), "hf_hub_token.txt")
    if os.path.exists(token_path):
        with open(token_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


HF_TOKEN = _load_hf_token()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 0


def set_seed(seed, determinism_level="medium"):
    """Set seed for random number generators with configurable determinism level.
    
    Args:
        seed: The random seed value
        determinism_level: One of "low", "medium", "high"
            - "low": Minimal seeding (only PyTorch manual_seed)
            - "medium": Basic reproducibility (PyTorch + CUDA seeding, cudnn deterministic)
            - "high": Full determinism (all RNGs, env vars, deterministic algorithms)
    """
    global SEED
    SEED = seed
    
    if determinism_level == "low":
        # Minimal seeding - only PyTorch
        torch.manual_seed(seed)
        return
    
    # Medium level: Basic PyTorch/CUDA seeding (matches good commit 02dba2b3)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # CuDNN algorithms may be non-deterministic. This can be fixed by setting:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if determinism_level == "medium":
        return
    
    # High level: Full determinism (aggressive seeding from bad commit 3d2ceca8)
    import random
    import numpy as np
    
    # Set seed for Python's built-in random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for Transformers library
    try:
        from transformers import set_seed as transformers_set_seed
        transformers_set_seed(seed)
    except ImportError:
        pass
    
    # Set environment variable for deterministic operations
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set additional deterministic flags for reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Enable deterministic algorithms in PyTorch (if available)
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            # Some operations don't have deterministic implementations
            pass


def empty_folder(folder_path):
    """
    Check if a folder is empty. If not, empty it.
    Args:
        folder_path (str): Path to the folder to check and empty.
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Check if the folder is empty
    if not os.listdir(folder_path):
        print(f"Folder '{folder_path}' is already empty.")
        return

    # Remove all contents of the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    print(f"Folder '{folder_path}' has been emptied.")


def get_folder_size(folder_path):
    """Returns the size of a folder in bytes, MB, and GB."""
    total_size = 0

    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.isfile(file_path):  # Ensure it's a file
                total_size += os.path.getsize(file_path)
    return total_size


def majority_vote(prototypes_list, distances_list, topk):
    """ Takes a list of lists containing selected prototypes and returns the most frequent ones. """
    if len(prototypes_list) == 1:
        return prototypes_list

    prototype_counts = Counter(prototypes_list)

    # If all three prototypes are different, select the one with the smallest distance
    if len(prototype_counts) == len(prototypes_list):
        min_index = distances_list.index(min(distances_list))
        return [prototypes_list[min_index]]

    # Extract top-k prototypes with the highest counts
    most_common = prototype_counts.most_common(topk)
    majority_prototypes = [p for p, _ in most_common]

    return majority_prototypes


class AdaptiveThreshold:
    """
    A class to initialize an adaptive threshold manager for the router based on the distribution of prototypes distances.
    """
    def __init__(self, default_threshold=0.1):
        self.default_threshold = default_threshold

    def update_threshold(self, idx_closest_prototype, dict_prototypes):
        """
            Compute the cosine distance between the closest prototype and all the others.
            Dynamically update the threshold with the minimum distance found.
            If only one prototype exists, return the default threshold.

            Args:
            - idx_closest_prototype: The index of the prototype closest to the new embedding.
            - dict_prototypes: Dictionary containing prototype embeddings.
            Return:
            - Updated threshold value.
        """
        if len(dict_prototypes) < 2:
            return self.default_threshold

        # Compute cosine distance
        cos_distances = {
            key: 1 - F.cosine_similarity(dict_prototypes[idx_closest_prototype[0]]["embedding"], prototype["embedding"], dim=1).item()
            for key, prototype in dict_prototypes.items()
        }

        min_key = min(cos_distances, key=cos_distances.get)
        min_value = cos_distances[min_key]
        return min_value / 2


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    SeeAlso:
        https://stackoverflow.com/a/7346105
        https://dev.to/vtsen/how-to-create-singleton-class-in-kotlin-5123

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


@Singleton
class Logger:

    def __init__(self, formatter='%(asctime)-2s # %(levelname)-2s # %(message)s'):
        self.formatter = logging.Formatter(formatter)

        # Primary logger for stdout + log.log
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.log_file_path = None

        # Separate logger for verbose detail lines (details.log)
        self.details_logger = logging.getLogger(__name__ + ".details")
        self.details_logger.setLevel(logging.DEBUG)
        self.details_logger.propagate = False
        self.details_file_path = None

        # Context for detail prefixes
        self._ctx = {"step": None, "epoch": None, "data": None}

        # handlers
        self.file_handler = None
        self.details_handler = None

        out_handler = logging.StreamHandler(sys.stdout)
        out_handler.setLevel(logging.DEBUG)
        out_handler.addFilter(lambda record: record.levelno == logging.DEBUG)
        out_handler.setFormatter(self.formatter)

        err_handler = logging.StreamHandler(sys.stderr)
        err_handler.setLevel(logging.WARNING)
        err_handler.setFormatter(self.formatter)

        self.logger.addHandler(out_handler)
        self.logger.addHandler(err_handler)

    def set_log_dir(self, log_dir):
        """Set the directory where the log files will be stored."""
        if self.log_file_path is None:  # Ensure it's only set once
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            # Main log
            self.log_file_path = os.path.join(log_dir, "log.log")
            self.file_handler = logging.FileHandler(self.log_file_path)
            self.file_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.file_handler)
            # Details log (verbose per-example debug)
            self.details_file_path = os.path.join(log_dir, "details.log")
            self.details_handler = logging.FileHandler(self.details_file_path)
            self.details_handler.setFormatter(self.formatter)
            self.details_logger.addHandler(self.details_handler)
        else:
            self.logger.warning("Log directory has already been set!")

    def reset_handlers(self):
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
        for handler in list(self.details_logger.handlers):
            self.details_logger.removeHandler(handler)
        self.logger.handlers.clear()
        self.details_logger.handlers.clear()
        self.file_handler = None
        self.details_handler = None
        self.log_file_path = None
        self.details_file_path = None
        self._ctx = {"step": None, "epoch": None, "data": None}

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def critical(self, msg: str) -> None:
        self.logger.critical(msg)

    # ---------------------------
    # Detail logging API
    # ---------------------------
    def set_context(self, step=None, epoch=None, data=None) -> None:
        if step is not None:
            self._ctx["step"] = step
        if epoch is not None:
            self._ctx["epoch"] = epoch
        if data is not None:
            self._ctx["data"] = data

    def detail(self, msg: str) -> None:
        prefix_parts = []
        if self._ctx.get("step") is not None:
            prefix_parts.append(f"step={self._ctx['step']}")
        if self._ctx.get("epoch") is not None:
            prefix_parts.append(f"epoch={self._ctx['epoch']}")
        if self._ctx.get("data") is not None:
            prefix_parts.append(f"data={self._ctx['data']}")
        prefix = ("[" + ", ".join(prefix_parts) + "] ") if prefix_parts else ""
        self.details_logger.debug(prefix + msg)





def cast_all_lora_to(model, dtype: torch.dtype) -> int:
    """Cast all LoRA adapter weights (A/B) to `dtype`. Returns number of tensors moved."""
    moved = 0
    for m in model.modules():
        if isinstance(m, LoraLayer):
            for A in m.lora_A.values():
                if A.weight.dtype != dtype:
                    A.weight.data = A.weight.data.to(dtype); moved += 1
            for B in m.lora_B.values():
                if B.weight.dtype != dtype:
                    B.weight.data = B.weight.data.to(dtype); moved += 1
    return moved

def clear_unsloth_cache(cache_dir=None):
    """Delete Unsloth compiled cache so kernels rebuild with the right dtype."""
    if cache_dir is None:
        # match your project layout
        cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "unsloth_compiled_cache"))
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)


def move_lora_adapters_device(model, active_adapter: str, device_active: str = "cuda", device_inactive: str = "cpu") -> tuple[int, int]:
    """
    Move LoRA adapter weights so that only the active adapter stays on `device_active`, and all
    inactive adapters are moved to `device_inactive`. Returns (moved_to_active, moved_to_inactive).
    This avoids deleting adapters while keeping GPU memory close to base + one LoRA.
    """
    moved_active = 0
    moved_inactive = 0
    for m in model.modules():
        if isinstance(m, LoraLayer):
            # lora_A / lora_B are dicts keyed by adapter name
            for name, A in getattr(m, "lora_A", {}).items():
                if hasattr(A, "weight"):
                    target_device = device_active if name == active_adapter else device_inactive
                    current_device = str(A.weight.device)
                    if current_device != target_device:
                        A.weight.data = A.weight.data.to(target_device)
                        if name == active_adapter:
                            moved_active += 1
                        else:
                            moved_inactive += 1
            for name, B in getattr(m, "lora_B", {}).items():
                if hasattr(B, "weight"):
                    target_device = device_active if name == active_adapter else device_inactive
                    current_device = str(B.weight.device)
                    if current_device != target_device:
                        B.weight.data = B.weight.data.to(target_device)
                        if name == active_adapter:
                            moved_active += 1
                        else:
                            moved_inactive += 1
    return moved_active, moved_inactive

# -----------------------------------------------------------------------------
# Backward-compatibility shim for removed debug function
# Some older code imports print_lora_dtypes from callm.utils. Provide a no-op
# so imports succeed without emitting verbose logs.
# -----------------------------------------------------------------------------

def print_lora_dtypes(model):
    """Compatibility shim: previously printed LoRA dtypes; now a no-op."""
    return


# -----------------------------------------------------------------------------
# Model wrapper diagnostics + mitigation
# -----------------------------------------------------------------------------


def format_wrapper_chain(chain, keep_head: int = 4, keep_tail: int = 1) -> str:
    """Format a wrapper chain for logs without exploding log size."""
    try:
        chain = list(chain or [])
    except Exception:
        return "<unavailable>"
    if not chain:
        return "<empty>"
    if len(chain) <= keep_head + keep_tail:
        return " -> ".join(chain)
    head = " -> ".join(chain[:keep_head])
    tail = " -> ".join(chain[-keep_tail:])
    return f"{head} -> ... -> {tail} (len={len(chain)})"


def diagnose_forward_wrappers(model, max_unwrap: int = 200) -> tuple[int, list[str]]:
    """Return (depth, chain) where depth counts nested forward wrappers.

    This is intended for diagnostics only (no mutation). We look for patterns
    commonly introduced by Accelerate and autocast wrappers.
    """
    fwd = getattr(model, "forward", None)
    chain: list[str] = []
    depth = 0
    seen = set()

    while fwd is not None and depth < max_unwrap:
        fid = id(fwd)
        if fid in seen:
            chain.append("<cycle>")
            break
        seen.add(fid)

        t = type(fwd)
        name = getattr(fwd, "__qualname__", None) or getattr(fwd, "__name__", None) or t.__name__
        chain.append(f"{t.__module__}.{t.__name__}({name})")

        if hasattr(fwd, "model_forward"):
            try:
                fwd = getattr(fwd, "model_forward")
                depth += 1
                continue
            except Exception:
                break

        if hasattr(fwd, "__wrapped__"):
            try:
                fwd = getattr(fwd, "__wrapped__")
                depth += 1
                continue
            except Exception:
                break

        break

    return depth, chain


def diagnose_module_wrappers(model, max_unwrap: int = 50) -> tuple[int, list[str]]:
    """Return (depth, chain) following `.module` wrappers (DDP-like)."""
    cur = model
    chain: list[str] = [type(cur).__name__]
    depth = 0
    while hasattr(cur, "module") and depth < max_unwrap:
        try:
            cur = cur.module
        except Exception:
            break
        depth += 1
        chain.append(type(cur).__name__)
    return depth, chain


def log_model_wrapper_diagnostics(model, label: str) -> dict:
    """Log wrapper-depth diagnostics without mutating state."""
    try:
        mid = id(model)
        mtype = type(model).__name__
        ftype = type(getattr(model, "forward", None)).__name__
        m_depth, m_chain = diagnose_module_wrappers(model)
        f_depth, f_chain = diagnose_forward_wrappers(model)
        Logger.instance().debug(
            f"[Diag][wrappers][{label}] model_id={mid} model_type={mtype} "
            f"module_depth={m_depth} module_chain={format_wrapper_chain(m_chain)} "
            f"forward_depth={f_depth} forward_type={ftype} forward_chain={format_wrapper_chain(f_chain)}"
        )
        return {
            "model_id": mid,
            "model_type": mtype,
            "module_depth": m_depth,
            "forward_depth": f_depth,
            "forward_type": ftype,
        }
    except Exception as e:
        Logger.instance().debug(f"[Diag][wrappers][{label}] diagnostics failed: {e}")
        return {}


def reset_forward_overrides(model) -> int:
    """Clear instance-level `forward` override(s) on the provided model.

    Accelerate/Trainer can wrap `model.forward` (e.g., convert_outputs_to_fp32 + autocast)
    and leave the wrapped callable on the model instance. Repeating this across many
    chunks creates a very deep wrapper chain and eventually triggers RecursionError.

    By deleting the instance attribute, Python falls back to the class-defined forward.

    Returns:
        Number of model objects where an instance-level `forward` attribute was removed.
    """
    cleared = 0
    try:
        targets = []
        seen = set()

        def _add(x):
            if x is None:
                return
            xid = id(x)
            if xid in seen:
                return
            seen.add(xid)
            targets.append(x)

        _add(model)
        # Also attempt common nested references (PEFT + wrappers)
        _add(getattr(model, "module", None))
        _add(getattr(model, "base_model", None))
        _add(getattr(model, "model", None))
        if hasattr(model, "get_base_model"):
            try:
                _add(model.get_base_model())
            except Exception:
                pass

        for t in targets:
            d = getattr(t, "__dict__", None)
            if isinstance(d, dict) and "forward" in d:
                try:
                    del d["forward"]
                    cleared += 1
                except Exception:
                    pass
    except Exception:
        # Never fail training due to cleanup helpers.
        return cleared

    return cleared
