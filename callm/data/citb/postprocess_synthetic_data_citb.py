import json
from tqdm import tqdm
import os
import argparse

def postprocess_data(data_path: str, output_path: str):

    # Read JSONL file (one JSON object per line)
    raw_data = []
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                raw_data.append(json.loads(line))

    processed_data = []
    seen = set()
    for entry in tqdm(raw_data):
        prompt = entry.get('prompt', '').strip()
        answer = entry.get('answer', '').strip()
        key = (prompt, answer)
        if key in seen:
            continue
        seen.add(key)
        processed_data.append(entry)

    print("Original size:", len(raw_data))
    print("Processed size:", len(processed_data))
    # Write output as JSONL
    with open(output_path, 'w') as f:
        for entry in processed_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def main(config):

    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.endswith('.json'):
            data_path = os.path.join(input_dir, fname)
            print("\nProcessing:", data_path)
            output_path = os.path.join(output_dir, fname)
            postprocess_data(data_path, output_path)


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, default="callm/data/citb/generated_data/CITB/v0.6")
    parser.add_argument("--output_dir", type=str, default="callm/data/citb/generated_data/CITB/v0.6_remove_duplicates")

    args = parser.parse_args()
    config = vars(args)

    return config

if __name__ == "__main__":
    config = get_config()
    main(config)