from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import gc
import os

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from callm.utils import Logger


@dataclass
class MemStats:
    allocated_gb: float
    reserved_gb: float


class MemoryManager:
    """
    Centralized memory utilities:
    - cleanup(): run gc + CUDA cache clears safely
    - report(): return current CUDA mem stats (if available)
    - adjust_batch_size(): shrink batch size if reserved/total exceeds threshold
    """

    @staticmethod
    def cleanup(label: Optional[str] = None) -> Optional[MemStats]:
        # Optional environment toggle to disable cleanup (for benchmarking)
        if os.getenv("CALLM_DISABLE_MEM_CLEANUP", "0") in {"1", "true", "TRUE"}:
            Logger.instance().debug(f"Memory cleanup disabled by env (CALLM_DISABLE_MEM_CLEANUP) for {label or ''}")
            return MemoryManager.report(label)
        try:
            gc.collect()
            if torch and getattr(torch, 'cuda', None) and torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'ipc_collect'):
                    torch.cuda.ipc_collect()
                return MemoryManager.report(label)
        except Exception as e:
            Logger.instance().debug(f"Memory cleanup{(' ['+label+']') if label else ''} failed: {e}")
        return None

    @staticmethod
    def report(label: Optional[str] = None) -> Optional[MemStats]:
        try:
            if torch and getattr(torch, 'cuda', None) and torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024 ** 3
                reserv = torch.cuda.memory_reserved() / 1024 ** 3
                stats = MemStats(allocated_gb=alloc, reserved_gb=reserv)
                if label:
                    Logger.instance().debug(
                        f"CUDA mem{(' ['+label+']') if label else ''}: allocated {alloc:.2f} GB | reserved {reserv:.2f} GB"
                    )
                return stats
        except Exception as e:
            Logger.instance().debug(f"Memory report{(' ['+label+']') if label else ''} failed: {e}")
        return None

    @staticmethod
    def adjust_batch_size(current_bs: int, threshold: float = 0.80) -> int:
        """
        If CUDA reserved / total exceeds threshold, iteratively halve batch size
        (down to 1) and return the new value. If CUDA is unavailable, returns current.
        Honor CALLM_DISABLE_BATCH_ADJUST to skip adjustments.
        """
        # Optional environment toggle to disable batch-size auto-adjust
        if os.getenv("CALLM_DISABLE_BATCH_ADJUST", "0") in {"1", "true", "TRUE"}:
            Logger.instance().debug("Batch-size auto-adjust disabled by env (CALLM_DISABLE_BATCH_ADJUST)")
            return int(current_bs)
        if not (torch and getattr(torch, 'cuda', None) and torch.cuda.is_available()):
            return current_bs
        try:
            bs = int(current_bs)
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory if props else 0
            if total <= 0:
                return bs
            ratio = torch.cuda.memory_reserved() / total
            while bs > 1 and ratio > threshold:
                bs = max(1, bs // 2)
                MemoryManager.cleanup("adjust-batch-size")
                ratio = torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory
            return bs
        except Exception:
            return current_bs
