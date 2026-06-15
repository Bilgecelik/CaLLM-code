# CaLLM

Camera-ready code release for CaLLM, a continual learning framework for adapting large language models with LoRA adapters and prototype-based routing.

The code supports:

- CALLM continual adapter training with prototype routing
- Inference-only evaluation
- Single-LoRA, EWC, and experience replay baselines
- TRACE and CITB-style task streams
- Accuracy, ROUGE-L, SARI, and EDIM evaluation

## Repository Layout

```text
.
├── main.py                         # Main experiment entrypoint
├── baselines/                      # Inference-only, single-LoRA, EWC, and ER baselines
├── callm/                          # Core routing, training, generation, and evaluation modules
│   └── data/
│       ├── trace/                  # TRACE stream loader
│       └── citb/                   # CITB stream and synthetic-data utilities
├── notebooks/                      # Minimal training example
├── scripts/                        # Optional helper scripts
└── requirements.txt
```

Large datasets, generated synthetic data, model checkpoints, adapter weights, logs, and private token files are intentionally not included in this public release.

## Installation

Create a fresh environment, then install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For gated Hugging Face models, authenticate with the Hugging Face CLI or provide a token through an environment variable:

```bash
export HF_TOKEN=...
```

W&B logging is disabled by default. To enable it, pass `--wandb_name <project>` and, optionally, `--wandb_project <entity>`.

## Data

This release contains code and small split/config files only. Prepare data locally in the following locations.

TRACE:

```text
callm/data/trace/LLM-CL-Benchmark_500/<task>/{train,eval,test}.json
callm/data/trace/LLM-CL-Benchmark_1000/<task>/{train,eval,test}.json
callm/data/trace/LLM-CL-Benchmark_5000/<task>/{train,eval,test}.json
```

Supported TRACE task names are:

```text
C-STANCE, FOMC, MeetingBank, Py150, ScienceQA, NumGLUE-cm, NumGLUE-ds, 20Minuten
```

CITB:

```text
callm/data/citb/tasks/<task_name>.json
callm/data/citb/generated_data/CITB/<version>/<task_name>.json  # optional synthetic data
```

CITB split files are included under `callm/data/citb/splits/`.

## Running Experiments

Inference-only TRACE evaluation:

```bash
python main.py \
  --baseline inference_only \
  --data_stream TRACE-500 \
  --datasets C-STANCE \
  --use_gt_metrics \
  --output_dir outputs/inference_trace500
```

CALLM on a TRACE stream:

```bash
python main.py \
  --baseline callm \
  --data_stream TRACE-500 \
  --stream_type Batched \
  --datasets ALL \
  --use_gt_metrics \
  --max_number_prototypes 4 \
  --output_dir outputs/callm_trace500
```

Online CALLM:

```bash
python main.py \
  --baseline callm \
  --data_stream TRACE-500 \
  --stream_type Online \
  --chunk_size 40 \
  --max_chunks 20 \
  --use_gt_metrics
```

Baselines:

```bash
python main.py --baseline train_single_lora --data_stream TRACE-500 --datasets C-STANCE --use_gt_metrics
python main.py --baseline ewc --data_stream TRACE-500 --datasets C-STANCE --use_gt_metrics
python main.py --baseline er --data_stream TRACE-500 --datasets C-STANCE --use_gt_metrics
```

Optional multi-run helpers live under `scripts/`.

Useful options:

- `--base_model`: Hugging Face or Unsloth model id
- `--prompt_style`: one of `alpaca`, `chat`, `simple`, `instruct`, `llama`, `gemma3`
- `--lora_r`, `--lora_alpha`, `--learning_rate`, `--num_epochs`
- `--gen_batch_size`, `--gen_max_new_tokens`
- `--debug`: quick TRACE debug configuration

## Outputs

Runs write logs, per-sample evaluation details, performance tables, and adapter checkpoints under `--output_dir`.

Output directories are ignored by git.

## License

This project is released under the Apache-2.0 license. See [LICENSE](LICENSE).
