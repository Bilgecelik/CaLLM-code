from __future__ import annotations
from typing import List, Optional, Tuple
import os
import torch
from peft import PeftModel, LoraConfig
from unsloth import FastModel
from callm.utils import Logger, move_lora_adapters_device, cast_all_lora_to, clear_unsloth_cache


class AdapterManager:
    """
    Central manager for LoRA adapter attachment, switching, merging, and
    device movement between active and inactive adapters.
    """
    def __init__(self):
        pass

    def safe_unload_if_supported(self, model):
        """Use Unsloth's unload() if available and model is not a PeftModel."""
        try:
            if not isinstance(model, PeftModel) and hasattr(model, "unload"):
                model = model.unload()
                Logger.instance().debug("Unloaded adapters from base model using unsloth unload().")
        except Exception as e:
            Logger.instance().debug(f"safe_unload_if_supported: skip unload ({e})")
        return model

    def attach_adapter(self, model, adapter_dir: str, adapter_name: Optional[str] = None):
        """
        Attach an adapter at adapter_dir to the given model (base or PeftModel).
        Always uses 'default' as adapter name for simplicity.
        Returns the possibly wrapped PeftModel.
        """
        adapter_name = "default"

        try:
            if not isinstance(model, PeftModel):
                model = PeftModel.from_pretrained(model, adapter_dir, adapter_name=adapter_name)
            else:
                _ = model.load_adapter(adapter_dir, adapter_name=adapter_name)
            model.set_adapter(adapter_name)

            # Cast LoRA weights to bf16 to match base precision if needed
            moved = cast_all_lora_to(model, torch.bfloat16)
            if moved:
                clear_unsloth_cache()

            Logger.instance().debug(f"Adapter '{adapter_name}' attached and activated.")
        except Exception as e:
            Logger.instance().critical(f"attach_adapter failed for {adapter_dir}: {e}")
            raise
        return model

    def set_active(self, model, adapter_name: str, device_active: Optional[str] = None, device_inactive: str = "cpu") -> Tuple[int, int]:
        """Activate adapter_name and move inactive adapters to CPU. Returns (moved_active, moved_inactive)."""
        if isinstance(model, PeftModel):
            model.set_adapter(adapter_name)
        try:
            if device_active is None:
                device_active = str(next(model.parameters()).device)
            act, inact = move_lora_adapters_device(model, adapter_name, device_active=device_active, device_inactive=device_inactive)
            return act, inact
        except Exception as e:
            Logger.instance().debug(f"set_active device move failed: {e}")
            return 0, 0

    def merge_adapters(self, model, adapter_dirs: List[str], weights: List[float], strategy: str = "ties", density: float = 0.5) -> str:
        """
        Load multiple adapters and create a weighted merged adapter on the model.
        Returns the merged adapter name.
        """
        adapter_names: List[str] = []
        for i, d in enumerate(adapter_dirs):
            name = os.path.basename(d)
            adapter_names.append(name)
            if i == 0:
                model = self.attach_adapter(model, d, adapter_name=name)
            else:
                _ = model.load_adapter(d, adapter_name=name)

        merged_name = "-".join(adapter_names)
        try:
            model.add_weighted_adapter(adapter_names, weights, merged_name, combination_type=strategy, density=density)
            model.set_adapter(merged_name)
            Logger.instance().debug(f"Merged adapters into '{merged_name}' with strategy={strategy}.")
        except Exception as e:
            Logger.instance().critical(f"merge_adapters failed: {e}")
            raise
        return merged_name

    def create_new_adapter(self, model, config: dict):
        """
        Create a new LoRA adapter for the given model.
        Always uses 'default' as adapter name for simplicity.
        - If model is already a PeftModel, add a new adapter via LoraConfig.
        - Otherwise, convert the base model to a PeftModel via Unsloth FastModel.get_peft_model.
        Returns the updated model.
        """
        adapter_name = "default"

        try:
            if isinstance(model, PeftModel):
                # Add a new adapter to existing PEFT model
                lora_config = LoraConfig(
                    r=config["lora_r"],
                    lora_alpha=config["lora_alpha"],
                    lora_dropout=config["lora_dropout"],
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                existing = getattr(model, "peft_config", {})
                if adapter_name in existing:
                    model.set_adapter(adapter_name)
                    Logger.instance().debug(f"Switched to existing adapter '{adapter_name}'.")
                else:
                    model.add_adapter(adapter_name, lora_config)
                    model.set_adapter(adapter_name)
                    Logger.instance().debug(f"New adapter '{adapter_name}' added to PEFT model.")
            else:
                # First-time LoRA creation on a plain base model via Unsloth helper
                model = FastModel.get_peft_model(
                    model,
                    r=config["lora_r"],
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_alpha=config["lora_alpha"],
                    lora_dropout=config["lora_dropout"],
                    lora_dtype=torch.bfloat16,
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                    random_state=3407,
                    use_rslora=False,
                    loftq_config=None,
                )
                Logger.instance().debug("New LoRA adapter created on base model.")

            # Cast to bf16 and cleanup Unsloth cache if we changed dtypes
            moved = cast_all_lora_to(model, torch.bfloat16)
            if moved:
                clear_unsloth_cache()

            return model
        except Exception as e:
            Logger.instance().critical(f"create_new_adapter failed: {e}")
            raise

    def get_active_adapter_name(self, model) -> Optional[str]:
        """Return the active adapter name for a PeftModel if available."""
        try:
            if isinstance(model, PeftModel):
                active = getattr(model, "active_adapter", None) or getattr(model, "active_adapters", None)
                if isinstance(active, list) and active:
                    return active[-1]
                if isinstance(active, str):
                    return active
        except Exception:
            pass
        return None

    def save_adapter(self, model, tokenizer, save_dir: str) -> None:
        """
        Save the active adapter weights and tokenizer into save_dir.
        Handles directory creation and logs result.
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            # Save model adapters (PeftModel) or full model fallback
            model.save_pretrained(save_dir)
            # Save tokenizer alongside
            tokenizer.save_pretrained(save_dir)
            name = self.get_active_adapter_name(model) or os.path.basename(save_dir)
            Logger.instance().debug(f"LoRA adapter '{name}' saved to: {save_dir}")
        except Exception as e:
            Logger.instance().critical(f"save_adapter failed for {save_dir}: {e}")
            raise
