import os
import psutil
import random
import torch
from torch.utils.data import Subset
from torch.utils.data._utils.collate import default_collate
import numpy as np
from transformers import BitsAndBytesConfig, AutoModelForMaskedLM, AutoTokenizer
from unsloth import FastLanguageModel as FastModel
from peft import PeftModel

from callm.prototypes_manager import PrototypeManager
from callm.utils import DEVICE, HF_TOKEN, AdaptiveThreshold, get_folder_size, Logger


class Router:
    """
        Determines which LoRa models to use for inference based on incoming data and current system state.
        Uses a decision-making system based on ProtoNet computed on the embedding space.
        Args:
            config: config arguments
            peft_registry: a registry of pre-trained PEFT model. If empty we'll train them from scratch.
    """
    def __init__(self, config, encoder_model, tokenizer, peft_registry=None):
        self.config = config

        if peft_registry:
            self.peft_registry = peft_registry
        else:
            self.peft_registry = set()

        # Reuse provided base model/tokenizer to avoid extra loads and VRAM
        self.encoder = encoder_model
        self.tokenizer = tokenizer

        # Ensure compatibility for embedding extraction across Unsloth/HF models
        if hasattr(self.encoder, "config"):
            try:
                self.encoder.config.output_hidden_states = True
            except Exception:
                pass
            try:
                self.encoder.config.use_cache = False
            except Exception:
                pass
        # Define padding token if missing (e.g., some Gemma tokenizers)
        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            try:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception:
                pass

        # Freeze the model weights for routing usage
        try:
            for param in self.encoder.parameters():
                param.requires_grad = False
        except Exception:
            pass
        # Ensure eval mode for deterministic behavior
        try:
            self.encoder.eval()
        except Exception:
            pass

        # Initialize the Prototype manager
        self.proto_manager = PrototypeManager()
        if config["max_number_prototypes"] is None:
            mem_info = psutil.virtual_memory()
            occupied_memory = get_folder_size(config["output_dir"])
            try:
                self.max_number_prototypes = (mem_info.available // occupied_memory) - 1
            except ZeroDivisionError:
                self.max_number_prototypes = 100
        else:
            self.max_number_prototypes = config["max_number_prototypes"]

        # Initialize the threshold manager
        if self.config["router_threshold"] is None:
            self.threshold_manager = AdaptiveThreshold()

    def extract_hidden_state(self, batch):
        """ Inspired by https://blog.min.io/feature-extraction-with-large-language-models-hugging-face-and-minio/. """
        text_field = batch['text']
        
        if text_field[0] is not None:
            Logger.instance().debug(f"[DEBUG] text_field[0] sample: {str(text_field[0])[:100]}")
        
        # Handle Gemma3 processor: extract inner tokenizer if tokenizer is a processor
        tokenizer_to_use = self.tokenizer
        if hasattr(self.tokenizer, 'tokenizer'):
            # Gemma3Processor has .tokenizer attribute for the actual tokenizer
            tokenizer_to_use = self.tokenizer.tokenizer
            Logger.instance().debug(f"[DEBUG] Using inner tokenizer from processor: {type(tokenizer_to_use)}")
        
        # assume batch is a dictionary with 'text' field containing the formatted prompts
        inputs = tokenizer_to_use(text_field, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # Ensure inputs are on the same device as the encoder/backbone (handles CPU offloading vs CUDA)
        enc_module = getattr(self.encoder, 'model', self.encoder)  # prefer backbone to avoid lm_head compute
        encoder_device = next(enc_module.parameters()).device
        inputs = {key: tensor.to(encoder_device) for key, tensor in inputs.items()}

        # Extract features (disable TorchDynamo to avoid Unsloth Gemma compile arg mismatches)
        with torch.no_grad():
            try:
                import torch._dynamo as _dynamo

                @_dynamo.disable
                def _enc_no_compile(model, **kwargs):
                    return model(**kwargs)

                if isinstance(self.encoder, PeftModel) and hasattr(self.encoder, "disable_adapter"):
                    # Call through PeftModel so disable_adapter actually takes effect and dtype/dev handling is preserved
                    with self.encoder.disable_adapter():
                        outputs = _enc_no_compile(
                            self.encoder,
                            **inputs,
                            output_hidden_states=True,
                            use_cache=False,
                            return_dict=True,
                        )
                else:
                    outputs = _enc_no_compile(
                        enc_module,
                        **inputs,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
            except Exception:
                # Fallback to normal forward if dynamo is unavailable
                if isinstance(self.encoder, PeftModel) and hasattr(self.encoder, "disable_adapter"):
                    with self.encoder.disable_adapter():
                        outputs = self.encoder(
                            **inputs,
                            output_hidden_states=True,
                            use_cache=False,
                            return_dict=True,
                        )
                else:
                    outputs = enc_module(
                        **inputs,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )

            # Prefer pooled last hidden states over logits for semantic similarity
            if getattr(outputs, "hidden_states", None) is not None and outputs.hidden_states:
                last_h = outputs.hidden_states[-1]  # [B, S, H]
                # Mean-pool over valid tokens using attention mask
                mask = inputs.get("attention_mask", None)
                if mask is None:
                    # Fallback: uniform mean over sequence
                    emb_per_sample = last_h.mean(dim=1)  # [B, H]
                else:
                    mask = mask.unsqueeze(-1).to(last_h.dtype)  # [B, S, 1]
                    summed = (last_h * mask).sum(dim=1)        # [B, H]
                    lengths = mask.sum(dim=1).clamp(min=1.0)   # [B, 1]
                    emb_per_sample = summed / lengths          # [B, H]
            else:
                # Fallback to logits if hidden states are unavailable (should be rare)
                logits = outputs.logits  # [B, S, V]
                emb_per_sample = logits.mean(dim=1)  # [B, V]

        # If you want a single batch-level embedding, average across the batch
        embedding = emb_per_sample.mean(dim=0, keepdim=True).cpu()

        return embedding

    def router_strategy(self, dataset, evaluation=False):
        """
        Decide which PEFT module to use for the current dataset.
        Return:
            selected_prototypes_dir: a list with the directory of the selected peft modules
            weights: list of values to merge the selected peft modules (if config.topk > 1)
        """
        sample_size = min(self.config['batch_size'], len(dataset))
        if sample_size == 0:
            raise ValueError("router_strategy received empty dataset for routing.")
        indices = random.sample(range(len(dataset)), sample_size)
        batch = default_collate([dataset[i] for i in indices])

        # Extract features for each batch
        embedding = self.extract_hidden_state(batch)
        # Find the closest prototypes. It returns the sorted indices and the relative distances
        selected_adapters, distances = self.proto_manager.find_closest_prototype(embedding, self.config["topk"])
        Logger.instance().debug(f"The closest prototype is: {selected_adapters}") 

        # Weights for merging adapters when topk > 1
        if len(selected_adapters) > 1:
            # Compute weights for each selected adapter
            # Normalize using a softmax
            weights = np.exp(distances)/sum(np.exp(distances))
        else:
            weights = [1.0]

        if not evaluation:
            if self.config["router_threshold"] is None:
                # Update the threshold dynamically
                threshold = self.threshold_manager.update_threshold(selected_adapters, self.proto_manager.prototypes)
                Logger.instance().debug(f"Threshold updated to {threshold}")
            else:
                threshold = self.config["router_threshold"]

            # If there is space for a new prototype, just create it
            if distances[0] > threshold and self.proto_manager.count < self.max_number_prototypes:
                # Create a new prototype
                selected_adapters = [self.proto_manager.count]
                self.peft_registry.add(self.proto_manager.count)
                self.proto_manager.add_prototype(embedding)

            # If there is no space for a new prototype, re-initialize the least frequent used and continue training
            elif distances[0] > threshold and self.proto_manager.count == self.max_number_prototypes:
                dict_prototypes = self.proto_manager.prototypes
                least_used_prototype = min(dict_prototypes.items(), key=lambda item: item[1]["usage_count"])
                self.proto_manager.reinitialize_prototype(least_used_prototype[0], embedding)

            # Update only the closest prototype
            else:
                self.proto_manager.update_prototype(selected_adapters[0], embedding, weights[0])

        # Return the dir of the selected adapters and the corresponding weights for merging
        selected_prototypes_dir = []
        for peft_dir in selected_adapters:
            if isinstance(peft_dir, int):
                # New prototype (integer ID)
                dir_name = str(peft_dir)
            else:
                # Existing prototype (string key)
                dir_name = peft_dir
            selected_prototypes_dir.append(os.path.join(self.config["output_dir"], dir_name))

        return selected_prototypes_dir, weights
