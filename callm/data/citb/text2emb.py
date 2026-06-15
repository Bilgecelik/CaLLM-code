"""
    Code for creating embeddings from data. Run with 'python -m callm.data.citb.text2emb'
    Modified from https://github.com/XMUDeepLIT/SSR/blob/main/custom/niv2-c012/text2emb.py.
    Hyperparameters for generation are taken from https://arxiv.org/abs/2403.01244.
"""

import os
import argparse
import torch
import json
import numpy as np
from unsloth import FastModel

from callm.data.citb.prepare_ni_data import prepare_stream_citb
from callm.utils import Logger

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(config):
    # Base model
    print("Load base model")
    model, tokenizer = FastModel.from_pretrained(
            model_name=config["base_model"],
            max_seq_length=512,
            load_in_4bit=True,
            load_in_8bit=False,  # Match gemma_test.py
            full_finetuning=False,  # Match gemma_test.py
            dtype=torch.bfloat16,  # Explicitly set model dtype to bfloat16
        )
    model.eval() 
    
    input_dir = config["input_path"]
    for fname in os.listdir(input_dir):
        if fname.endswith('.json'):
            data_path = os.path.join(input_dir, fname)

            fname_npy = fname.replace('.json', '.npy')
            output_path = os.path.join(config['output_path'], fname_npy)
            if os.path.exists(output_path):
                print(f"Skip existing task: {fname_npy}")
                continue
            else:
                # Read JSONL file (one JSON object per line)
                task_data = []
                with open(data_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            task_data.append(json.loads(line)['text'])

                print(f"Processing task: {fname} with {len(task_data)} examples")
            
                embed_list = None
                for i in range(0, len(task_data), config["batch_size"]):
                    batch = task_data[i:i+config["batch_size"]]
                    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    # Ensure inputs are on the same device as the encoder/backbone (handles CPU offloading vs CUDA)
                    model_device = next(model.parameters()).device
                    inputs = {key: tensor.to(model_device) for key, tensor in inputs.items()}

                    # Extract features (disable TorchDynamo to avoid Unsloth Gemma compile arg mismatches)
                    with torch.no_grad():
                        try:
                            import torch._dynamo as _dynamo

                            @_dynamo.disable
                            def _enc_no_compile(model, **kwargs):
                                return model(**kwargs)
                            
                            outputs = _enc_no_compile(
                                model,
                                **inputs,
                                output_hidden_states=True,
                                use_cache=False,
                                return_dict=True,
                            )
                        except Exception:
                            # Fallback to normal forward if dynamo is unavailable
                                outputs = model(
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
                    embeddings = emb_per_sample.mean(dim=0, keepdim=True).cpu()
                    embed_list = embeddings if embed_list is None else torch.cat([embed_list, embeddings], dim=0)

                print(f"emb_list.shape:{embed_list.shape}")
                embed_list = embed_list.float().numpy()

                os.makedirs(config['output_path'], exist_ok=True)
                fname_npy = fname.replace('.json', '.npy')
                np.save(os.path.join(config["output_path"], fname_npy), embed_list)


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_stream", type=str, default="CITB")
    parser.add_argument("--input_path", type=str, default="callm/data/citb/generated_data/CITB/v0.6_remove_duplicates")
    parser.add_argument("--output_path", type=str, default="callm/data/citb/generated_data/CITB/text2emb_v0.6_new")
    parser.add_argument("--base_model", type=str, default="unsloth/gemma-3-12b-it-unsloth-bnb-4bit", choices=["unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", "unsloth/gemma-3-12b-it-unsloth-bnb-4bit"])
    parser.add_argument("--load_in_4bit", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()
    config = vars(args)

    return config

if __name__ == "__main__":
    config = get_config()
    main(config)
