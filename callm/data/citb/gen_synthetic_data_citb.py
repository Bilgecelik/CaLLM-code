"""
    Code for synthetic examples generation. Run with "python -m callm.data.citb.gen_synthetic_data_citb".
    Modified from https://github.com/XMUDeepLIT/SSR/blob/main/custom/icl_gen/complete_param_nic010_cate.py.
    Hyperparameters for generation are taken from https://arxiv.org/abs/2403.01244.
"""

import os
import argparse
import torch
import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
from tqdm import tqdm
import jsonlines

from callm.data.citb.prepare_ni_data import prepare_stream_citb
from callm.utils import Logger



INSTRUCTION_MARKER = "### Instruction:\n"
RESPONSE_MARKER = "### Response:\n"


def times(n:int, length:int) -> List[List[int]]:
    '''
    n: number of shots.
    length: length of 
    '''
    if n == 1: return [[i] for i in range(length)]
    res_list = []
    for i in range(length):
        for lis in times(n-1, length):
            if i not in lis:
                res_list.append([i]+lis)
    return res_list


def prepare_instructions(tokenizer, data, n_shots, permutations, max_length):
    all_perm_list = times(n_shots, permutations)  # generate a list of permutations of length n_shots from the pool size
    inst_list = []
    for lis in all_perm_list:
        instruction = ""
        for i in lis:
            instruction += (data[i]["text"] + "\n\n")  # data in alpaca format
        if len(tokenizer.tokenize(instruction))>=max_length:
            continue
        else:
            inst_list.append(instruction)
    return inst_list


def postprocess_data(sequence, instruction):
    text = sequence[0]["generated_text"]
    generated_output_only = text.replace(instruction, '', 1).strip()

    instruction_start = generated_output_only.find(INSTRUCTION_MARKER) + len(INSTRUCTION_MARKER)
    response_start = generated_output_only.find(RESPONSE_MARKER) + len(RESPONSE_MARKER)

    prompt = generated_output_only[instruction_start:generated_output_only.find(RESPONSE_MARKER)].strip()
    answer = generated_output_only[response_start:].strip()

    data_dict = {
            "prompt": prompt,
            "answer": answer,
            "text": generated_output_only,
        }
    
    return data_dict


def icl_generation(pipeline, tokenizer, data, n_shots, permutations, max_length, output_path):
    inst_list = prepare_instructions(tokenizer, data, n_shots, permutations, max_length)

    for i, instruction in tqdm(enumerate(inst_list), total=len(inst_list)):
        sequence = pipeline(
                instruction,
                do_sample=True,
                temperature=0.9,
                max_new_tokens=512, # max tokens to generate
                num_beams=1,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                batch_size=1,
            )

        data_dict = postprocess_data(sequence, instruction)
        with jsonlines.open(output_path, "a") as file:
                file.write(data_dict)


def main(config):
    # Base model
    Logger.instance().info("Load base model")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    tokenizer.padding_side = "left" # to avoid RuntimeError: p.attn\_bias\_ptr is not correctly aligned with Gemma
    tokenizer.pad_to_multiple_of = 8 # to avoid RuntimeError: p.attn\_bias\_ptr is not correctly aligned with Gemma

    model = AutoModelForCausalLM.from_pretrained(config["base_model"], torch_dtype=torch.bfloat16)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    pipeline.model.config.use_cache = True
    pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set output path
    output_path = os.path.join("callm/data/citb/generated_data/", config["data_stream"], "v0.6")  # specify version here or config["output_path"] 
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Input data
    Logger.instance().info("Load input data")
    config_path = "callm/data/citb/ni_data_config.json"
    stream, task_names = prepare_stream_citb(config["data_stream"], config_path, tokenizer.eos_token)

    # Create synthetic data
    for task_name, task_data in zip(task_names, stream):
        output_filename = os.path.join(output_path, f"{task_name}.json")
        if os.path.exists(output_filename):
            print(f"Skip existing task: {task_name}")
            continue
        else:
            print(f"Generate synthetic data for task: {task_name}")
            train_data = task_data["train"]
            # Use the first example for extracting the task definition 
            # (assuming it is the same for all examples in the task)
            definition = train_data[0]['prompt'].split('\n\n')[0]
            output_filename = os.path.join(output_path, f"{task_name}.json")
            icl_generation(pipeline, tokenizer, train_data, config["n_shots"], config["permutations"], config["max_seq_length"], output_filename)



def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_stream", type=str, default="CITB")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--base_model", type=str, default="unsloth/gemma-3-12b-it-unsloth-bnb-4bit")
    parser.add_argument("--load_in_4bit", type=bool, default=True)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--n_shots", type=int, default=2)
    parser.add_argument("--permutations", type=int, default=80)

    args = parser.parse_args()
    config = vars(args)

    return config

if __name__ == "__main__":
    config = get_config()
    main(config)