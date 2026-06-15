import os
import sys
import shutil
import wandb
import argparse
import math
from callm.training_manager import TrainingManager
from callm.utils import Logger, set_seed
from callm.data.trace.prepare_trace_data import TRACE_METRICS, TRACE_METRICS_DEBUG

from baselines.simple_baselines import InferenceOnly, SingleLora
from baselines.ewc import EWC
from baselines.er import ER


def get_dataset_config(dataset_arg, data_stream):
    """Get datasets and evaluation metrics based on user input.
    
    Args:
        dataset_arg: Either a single dataset name or "ALL" for all datasets
        data_stream: The data stream being used (e.g., TRACE-500, TRACE-1000, TRACE-5000)
        
    Returns:
        tuple: (datasets_list, evaluation_metrics_list)
    """
    # Handle different data stream types
    if data_stream.startswith("TRACE-"):
        # TRACE dataset mapping
        trace_datasets_map = {
            "C-STANCE": "accuracy",
            "FOMC": "accuracy", 
            "MeetingBank": "rouge",
            "Py150": "edim",
            "ScienceQA": "accuracy",
            "NumGLUE-cm": "accuracy",
            "NumGLUE-ds": "accuracy",
            "20Minuten": "sari"
        }
        
        if dataset_arg.upper() == "ALL":
            # Return all TRACE datasets in original order
            datasets = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
            metrics = [trace_datasets_map[dataset] for dataset in datasets]
            return datasets, metrics
        else:
            # Single TRACE dataset
            if dataset_arg in trace_datasets_map:
                return [dataset_arg], [trace_datasets_map[dataset_arg]]
            else:
                available = list(trace_datasets_map.keys())
                raise ValueError(f"TRACE dataset '{dataset_arg}' not found. Available TRACE datasets: {available}")
    elif data_stream.startswith("CITB"):
        # For CITB data streams, datasets will be loaded from task split files
        # Return empty lists - will be populated later in main()
        return [], []
    else:
        # For other data streams, we can't provide automatic mapping
        # User must provide datasets and metrics manually through config or other means
        raise ValueError(f"Dataset filtering with --datasets parameter is only supported for TRACE data streams (TRACE-500, TRACE-1000, TRACE-5000) and CITB data streams (CITB, CITB-38). Current data stream: {data_stream}. For custom data streams, please modify the config directly.")

def parse_args():
    parser = argparse.ArgumentParser(description='CALLM - Continual AutoML Library for Large Language Models')
    parser.add_argument('--debug', action='store_true', default=False, 
                        help='Enable debug mode for testing')
    parser.add_argument('--baseline', type=str, default='inference_only',
                        choices=['callm', 'inference_only', 'train_single_lora', 'ewc', 'er'],
                        help='Baseline method to use')
    parser.add_argument('--base_model', type=str, default='unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',
                        help='Base model to use')
    parser.add_argument('--data_stream', type=str, default='TRACE-5000',
                        choices=['TRACE-500', 'TRACE-1000', 'TRACE-5000', 'CITB', 'CITB-38'],
                        help='Dataset size to use')
    parser.add_argument('--stream_type', type=str, default='Batched',
                        choices=['Online', 'Batched'],
                        help='Stream processing type')
    parser.add_argument('--use_gt_metrics', action='store_true', default=False,
                        help='Use ground truth metrics instead of auto metric selection')
    parser.add_argument('--max_number_prototypes', type=int, default=None,
                        help='Maximum number of prototypes (LoRA adapters). If None, depends on available memory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size (number of steps per optimizer step)')
    # Tunable HPs
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank (default: 16)')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate (default: 0.0001)')
    parser.add_argument('--num_epochs', type=int, default=5, help='Training epochs per chunk/task (default: 5)')
    parser.add_argument('--chunk_size', type=int, default=40, help='Online stream chunk size (default: 40)')
    parser.add_argument('--max_chunks', type=int, default=200, help='Maximum number of chunks to process in Online mode (default: 200)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps (default: 4)')
    parser.add_argument('--prompt_style', type=str, default='alpaca',
                        choices=['alpaca', 'chat', 'simple', 'instruct', 'llama', 'gemma3'],
                        help='Prompt template style to use (default: alpaca)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Temperature for text generation. If omitted, uses the model default.')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine',
                        choices=['constant', 'linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'inverse_sqrt'],
                        help='Learning rate scheduler type (default: cosine)')
    parser.add_argument('--load_best_model_at_end', action='store_true', default=False,
                        help='Load the best model checkpoint at end of training (default: False)')
    parser.add_argument('--early_stopping', action='store_true', default=False,
                        help='Enable early stopping during training (default: False)')
    parser.add_argument('--determinism_level', type=str, default='medium',
                        choices=['low', 'medium', 'high'],
                        help='Level of randomness control: low (minimal), medium (good run), high (full determinism)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Global RNG seed for training/evaluation (default: 0)')
    parser.add_argument('--stream_seed', type=int, default=42,
                        help='Seed for Online stream chunking/interleaving (default: 42)')
    parser.add_argument('--gen_batch_size', type=int, default=None,
                        help='Generation batch size for evaluation (default: same as batch_size)')
    parser.add_argument('--gen_max_new_tokens', type=int, default=None,
                        help='Maximum new tokens to generate during evaluation (default: metric-dependent)')
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='Number of training steps between evaluations (default: 100)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay for optimizer (default: 0.0)')
    parser.add_argument('--datasets', type=str, default='ALL',
                        help='Dataset name (single) or "ALL" for all datasets (default: ALL)')
    parser.add_argument('--max_seq_length', type=int, default=1024,
                        help='Maximum sequence length for tokenization (default: 1024)')
    parser.add_argument('--synthetic_data_path', type=str, default=None,
                        help='Path to synthetic data for CITB datasets (optional)')
    # Wandb arguments
    parser.add_argument('--wandb_name', type=str, default=None, 
                        help='Name of the project. If None wandb is offline')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Optional W&B entity/team name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='If set to None and wandb_name is set then the run name is set by default')
    parser.add_argument('--output_dir', type=str, default='outputs/',
                        help='Base output directory for experiment results (default: outputs/)')
    return parser.parse_args()

def generate_run_summary(config):
    """Generate a formatted run summary with key configuration parameters."""
    try:
        baseline = config.get('baseline')
        
        if baseline == 'inference_only':
            # Simplified summary for inference only
            temp_info = f"temp={config.get('temperature')}" if config.get('temperature') is not None else "temp=model_default"
            gen_batch = config.get('gen_batch_size', config.get('batch_size', 'auto'))
            
            summary = (
                "\n=== INFERENCE-ONLY RUN SUMMARY ================\n"
                f"Baseline: {baseline} | Stream: {config.get('stream_type')} | Data: {config.get('data_stream')}\n"
                f"Model: {config.get('base_model')}\n"
                f"Prompt Style: {config.get('prompt_style')} | {temp_info} | gen_batch={gen_batch}\n"
                f"Max Seq Length: {config.get('max_seq_length')}\n"
                f"Datasets: {', '.join(config.get('datasets', ['N/A']))}\n"
                f"Output: {config.get('output_dir')}\n"
                f"W&B: project={config.get('wandb_project')} | entity={config.get('wandb_entity')}\n"
                "===============================================\n"
            )
        else:
            # Full summary for training baselines
            bs = int(config.get("batch_size", 1))
            gas = int(config.get("gradient_accumulation_steps", 1))
            eff_batch = bs * gas
            epochs = config.get("num_epochs", 1)
            mode = str(config.get("stream_type", "Batched")).lower()
            
            if mode == "online":
                chunk = int(config.get("chunk_size", 1))
                steps_per_chunk = max(1, math.ceil(chunk / max(1, bs)))
                opt_steps_per_chunk = max(1, math.ceil(steps_per_chunk / max(1, gas)))
                cadence = f"steps/chunk={steps_per_chunk}, opt_steps/chunk={opt_steps_per_chunk}, max_chunks={config.get('max_chunks', 'n/a')}"
            else:
                cadence = "steps/epoch ~= ceil(N_train/batch_size), opt_steps/epoch ~= ceil(steps/epoch / grad_accum)"

            summary = (
                "\n=== TRAINING RUN SUMMARY =======================\n"
                f"Baseline: {baseline} | Stream: {config.get('stream_type')} | Data: {config.get('data_stream')}\n"
                f"Model: {config.get('base_model')} | Prompt: {config.get('prompt_style')}\n"
                f"LoRA: r={config.get('lora_r')}, alpha={config.get('lora_alpha')}, dropout={config.get('lora_dropout')}\n"
                f"Train: batch={bs}, grad_accum={gas}, effective_batch(per-device)={eff_batch}, lr={config.get('learning_rate')}, epochs={epochs}\n"
                f"Cadence: {cadence}\n"
                f"Output: {config.get('output_dir')}\n"
                f"W&B: project={config.get('wandb_project')} | entity={config.get('wandb_entity')}\n"
                "===============================================\n"
            )
        
        # Log to both terminal and log file using DEBUG level
        Logger.instance().debug(summary)
        
    except Exception as e:
        Logger.instance().warning(f"Could not print run summary: {e}")

def get_output_dir_name(config, base_output_dir):
    """Generate output directory name based on configuration parameters.
    
    For inference_only baseline: Uses cleaner format with model, prompt_style, temperature, and dataset info.
    For other baselines: Uses full format with training parameters.
    """
    # Extract model name from full path (e.g., 'Meta-Llama-3.1-8B-Instruct-bnb-4bit' from full path)
    model_name = config["base_model"].split('/')[-1] if '/' in config["base_model"] else config["base_model"]
    
    # Generate dataset part of the folder name
    datasets = config.get("datasets", [])
    if len(datasets) > 1:
        # Multiple datasets (ALL case)
        dataset_part = "datasets_all"
    elif len(datasets) == 1:
        # Single dataset - use lowercase and replace spaces/special chars with underscores
        dataset_name = datasets[0].lower().replace("-", "_").replace(" ", "_")
        dataset_part = f"datasets_{dataset_name}"
    else:
        # No datasets specified
        dataset_part = ""

    # Different folder naming for inference_only vs training baselines
    if config.get('baseline') == 'inference_only':
        # Clean format for inference only: baseline_model_data_stream_datasets_prompt_temp
        folder_parts = [
            f"{config['baseline']}",
            f"{model_name}",
            f"{config['data_stream']}",
            f"seed_{config.get('seed', 0)}",
            dataset_part,
            f"prompt_{config.get('prompt_style', 'alpaca')}",
        ]
        
        # Add temperature if specified
        if config.get('temperature') is not None:
            folder_parts.append(f"temp_{config['temperature']}")
            
        # Add stream type only if it's online (batched is default)
        if config['stream_type'].lower() == 'online':
            folder_parts.append(f"{config['stream_type'].lower()}")
            
    else:
        # Full format for training baselines
        folder_parts = [
            f"{config['baseline']}",
            f"{model_name}",
            f"{config['data_stream']}",
            f"{config['stream_type'].lower()}",
            dataset_part,
            f"seed_{config.get('seed', 0)}",
            f"batch_{config['batch_size']}",
            f"lora_r_{config['lora_r']}",
            f"lora_alpha_{config['lora_alpha']}",
            f"lr_{config['learning_rate']}",
            f"gracc_{config['gradient_accumulation_steps']}",
            f"epochs_{config['num_epochs']}",
            f"chunk_size_{config['chunk_size']}",
        ]
        
        # Add max_chunks only for Online mode
        if config['stream_type'].lower() == 'online':
            folder_parts.append(f"max_chunks_{config['max_chunks']}")

    # Common flags for all baselines
    if config.get("debug"):
        folder_parts.append("debug")

    if config.get("use_gt_metrics"):
        folder_parts.append("gt-metrics")

    if config.get("max_number_prototypes") is not None:
        folder_parts.append(f"proto-{config['max_number_prototypes']}")

    # Filter out empty parts and join
    folder_parts = [part for part in folder_parts if part and part != ""]
    folder_name = "_".join(str(p) for p in folder_parts)
    return os.path.join(base_output_dir, folder_name)

def main(config, run_id=None):
    if config["debug"]:
        # Small multi-task debug run with simple accuracy metrics
        config["datasets"] = ["NumGLUE-ds", "ScienceQA", "C-STANCE"]
        config["evaluation_metrics"] = ["accuracy", "accuracy", "accuracy"]
        config["data_stream"] = "TRACE-500"  # Use smaller dataset for debug
        config["output_dir"] = f"outputs_debug/callm_{run_id}"
        config["batch_size"] = 4  # smaller debug batch to avoid OOM with large vocabs
        config["chunk_size"] = 8  # Small chunk for debug
        config["max_chunks"] = 1  # Only 1 chunk for focused testing
        config["num_epochs"] = 0.001  # Very small for quick debug
        config["wandb"] = "none"
        config["max_seq_length"] = 512


    set_seed(config["seed"], config.get("determinism_level", "medium"))
    # Configure W&B routing
    if config.get("wandb_name"):
        os.environ["WANDB_PROJECT"] = config["wandb_name"]
    if config.get("wandb_project"):
        os.environ["WANDB_ENTITY"] = config["wandb_project"]

    if "callm" in config["baseline"]:
        # Run CALLM
        # Note: output_dir is already set to the full path in the main block

        #  Delete old output directory and create a new one
        if os.path.exists(config["output_dir"]):
            shutil.rmtree(config["output_dir"])  # Delete the directory and its contents
        os.makedirs(config["output_dir"])

        Logger.instance().set_log_dir(config["output_dir"])
        
        # ---- Run summary (before training starts) ----
        generate_run_summary(config)
        Logger.instance().debug(f"CALLM model saved in {config['output_dir']}")
        Logger.instance().debug(f"Detailed per-sample evaluation logs will be written to {os.path.join(config['output_dir'], 'details.log')}")
        trainer = TrainingManager(config)
        return trainer.train_continual()

    elif "inference_only" in config["baseline"]:
        # Run inference_only baseline
        # Note: output_dir is already set to the full path in the main block

        #  Delete old output directory and create a new one
        if os.path.exists(config["output_dir"]):
            shutil.rmtree(config["output_dir"])  # Delete the directory and its contents
        os.makedirs(config["output_dir"])

        Logger.instance().set_log_dir(config["output_dir"])
        
        # ---- Run summary (before inference starts) ----
        generate_run_summary(config)
        Logger.instance().debug(f"Inference only with model {config['base_model']} saved in {config['output_dir']}")
        Logger.instance().debug(f"Detailed per-sample evaluation logs will be written to {os.path.join(config['output_dir'], 'details.log')}")
        trainer = InferenceOnly(config)
        trainer.eval_single_task()

    elif "train_single_lora" in config["baseline"]:
        # Consider each task separately and train a single LoRA for each task
        # Note: output_dir is already set to the full path in the main block

        #  Delete old output directory and create a new one
        if os.path.exists(config["output_dir"]):
            shutil.rmtree(config["output_dir"])  # Delete the directory and its contents
        os.makedirs(config["output_dir"])

        Logger.instance().set_log_dir(config["output_dir"])
        
        # ---- Run summary (before training starts) ----
        generate_run_summary(config)
        Logger.instance().debug(f"Train a single LoRA per task with model {config['base_model']} saved in {config['output_dir']}")
        Logger.instance().debug(f"Detailed per-sample evaluation logs will be written to {os.path.join(config['output_dir'], 'details.log')}")
        trainer = SingleLora(config)
        trainer.train_single_lora()

    elif "ewc" in config["baseline"]:
        # Apply EWC to a continually trained LoRA
        # Note: output_dir is already set to the full path in the main block
        config["ewc_lambda"] = 1.8

        #  Delete old output directory and create a new one
        if os.path.exists(config["output_dir"]):
            shutil.rmtree(config["output_dir"])  # Delete the directory and its contents
        os.makedirs(config["output_dir"])

        Logger.instance().set_log_dir(config["output_dir"])
        
        # ---- Run summary (before training starts) ----
        generate_run_summary(config)
        Logger.instance().debug(f"EWC applied on a continually trained LoRA with model {config['base_model']} saved in {config['output_dir']}")
        Logger.instance().debug(f"Detailed per-sample evaluation logs will be written to {os.path.join(config['output_dir'], 'details.log')}")
        trainer = EWC(config)
        trainer.train_ewc()

    elif "er" in config["baseline"]:
        # Apply ER with a continually trained LoRA by storing a few old training samples with a buffer
        # Note: output_dir is already set to the full path in the main block
        config["buffer_size"] = 250

        #  Delete old output directory and create a new one
        if os.path.exists(config["output_dir"]):
            shutil.rmtree(config["output_dir"])  # Delete the directory and its contents
        os.makedirs(config["output_dir"])

        Logger.instance().set_log_dir(config["output_dir"])
        
        # ---- Run summary (before training starts) ----
        generate_run_summary(config)
        Logger.instance().debug(f"Experience replay with a continually trained LoRA with model {config['base_model']} saved in {config['output_dir']}")
        Logger.instance().debug(f"Detailed per-sample evaluation logs will be written to {os.path.join(config['output_dir'], 'details.log')}")
        trainer = ER(config)
        trainer.train_er()

    else:
        raise NotImplementedError("This baseline is not implemented yet")

def build_config(args):
    """Build an experiment configuration from parsed CLI arguments."""
    datasets, evaluation_metrics = get_dataset_config(args.datasets, args.data_stream)

    if args.data_stream.startswith("CITB"):
        data_split = "cl_dialogue_tasks.txt" if args.data_stream == "CITB" else "cl_38_random_tasks.txt"
        split_file_path = f"./callm/data/citb/splits/CIT_splits/{data_split}"
        try:
            with open(split_file_path, "r") as f:
                datasets = [line.strip() for line in f if line.strip()]
            evaluation_metrics = ["rouge"] * len(datasets)
        except FileNotFoundError:
            raise FileNotFoundError(f"CITB split file not found: {split_file_path}. Make sure CITB data is properly set up.")

    base_output_dir = os.path.expanduser(args.output_dir)

    config = {
        "seed": args.seed,
        "stream_seed": args.stream_seed,  # Online stream seed (default: 42)
        "debug": args.debug,  # From command line
        "baseline": args.baseline,  # From command line
        "base_model": args.base_model,  # From command line
        "load_in_4bit": True,
        "max_seq_length": args.max_seq_length,  # From command line (default: 1024)
        "temperature": args.temperature,  # From command line (default: 1.0)
        "data_stream": args.data_stream,  # From command line
        "stream_type": args.stream_type,  # From command line
        "datasets": datasets,  # From parsed datasets argument
        "evaluation_metrics": evaluation_metrics,  # From parsed datasets argument
        "output_dir": None,
        "batch_size": args.batch_size,  # From command line
        "chunk_size": args.chunk_size,  # From command line (per-user default 40)
        "max_chunks": args.max_chunks,  # From command line (default: 200)
        # Online-friendly defaults; can be overridden via CLI or config
        "num_epochs": args.num_epochs,  # From command line (per-user default 3)
        "learning_rate": args.learning_rate,  # From command line (per-user default 0.003)
        "weight_decay": args.weight_decay,  # From command line (default: 0.0)
        "use_dora": False,
        # eval/save/logging aligned with eval_steps from command line
        "eval_step": args.eval_steps,  # From command line (default: 100)
        "save_step": args.eval_steps,  # Aligned with eval_steps
        "logging_steps": args.eval_steps,  # Aligned with eval_steps
        "lora_r": args.lora_r,        # From command line (default 8)
        "lora_alpha": args.lora_alpha, # From command line (default 16)
        "lora_dropout": 0.05,
        # Keep default optimizer consistent across baselines.
        # NOTE: CALLM's SFTConfig in callm/lora_trainer.py uses optim="adamw_8bit".
        "optimizer": "adamw_8bit",
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": args.lr_scheduler_type,  # From command line (default: constant)
        "load_best_model_at_end": args.load_best_model_at_end,  # From command line (default: False)
        "early_stopping": args.early_stopping,  # From command line (default: False)
        "checkpoint_path": None,
        "encoder_model": args.base_model,
        "max_number_prototypes": args.max_number_prototypes,  # From command line
        "router_threshold": None,  # a number between [0,1] to select when to initialize a new prototype. If None dynamic assegnation
        "prompt_style": args.prompt_style,  # From command line (default: alpaca)
        "topk": 1,  # if more than 1, adapter_merge should be set to an option other than None
        "adapter_merge": None,  # possible options: #none" = single lora, top-k should be 1,or merge with "ties", "dare", ..
        "use_gt_metrics": args.use_gt_metrics,  # From command line - use ground truth metrics instead of auto selection
        "wandb_name": args.wandb_name,  # From command line - project name
        "wandb_project": args.wandb_project,  # From command line - entity
        "wandb_run_name": args.wandb_run_name,  # From command line - run name
        "wandb_entity": args.wandb_project,  # Keep for compatibility
        "determinism_level": args.determinism_level,  # From command line (default: medium)
        # Generation batch size - defaults to training batch_size if not specified
        "gen_batch_size": args.gen_batch_size if args.gen_batch_size is not None else args.batch_size,
        "gen_max_new_tokens": args.gen_max_new_tokens,  # None by default, allows metric-based defaults
        "synthetic_data_path": args.synthetic_data_path,  # Path to synthetic data for CITB
    }

    # Build output directory name now that config is defined
    dynamic_output_dir = get_output_dir_name(config, base_output_dir)
    config["output_dir"] = dynamic_output_dir

    # Handle debug mode dataset overrides
    if config["debug"]:
        if "TRACE" not in config["data_stream"]:
            Logger.instance().error("Use TRACE data for debugging")
            sys.exit()

        config["datasets"] = list(TRACE_METRICS_DEBUG.keys())
        config["evaluation_metrics"] = [TRACE_METRICS_DEBUG[ds] for ds in config["datasets"]]
        config["batch_size"] = 15
        config["chunk_size"] = 20
        config["max_chunks"] = 3
        config["num_epochs"] = 0.005

    # Data handlers for full datasets (if not debug mode)
    elif "TRACE" in config["data_stream"] and not datasets:  # if no specific datasets were selected
        config["datasets"] = list(TRACE_METRICS.keys())
        config["evaluation_metrics"] = [TRACE_METRICS[ds] for ds in config["datasets"]]

    return config


def get_config():
    """Return the default CLI-derived config."""
    return build_config(parse_args())


def init_wandb_if_enabled(config):
    """Initialize W&B only when --wandb_name is provided."""
    if config.get("wandb_name") is None:
        return

    folder_name = os.path.basename(config["output_dir"])
    if config.get("wandb_run_name") is None:
        config["wandb_run_name"] = folder_name

    wandb.init(
        project=config["wandb_name"],
        name=config["wandb_run_name"],
        entity=config.get("wandb_project"),
        config=config,
    )


if __name__ == "__main__":
    config = get_config()
    init_wandb_if_enabled(config)
    main(config)

    if config.get("wandb_name") is not None:
        wandb.finish()
