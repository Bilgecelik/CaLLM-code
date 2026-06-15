import os
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from callm.data.trace.prepare_trace_data import prepare_task_incremental_stream, init_dataloader
from callm.router import Router
from callm.utils import Logger
from unsloth import FastLanguageModel as FastModel


def main(config):

    def load_all_embeddings(datasets, output_dir):
        """Load all embeddings from saved .pt files."""
        embeddings_dict = {}

        for dataset_id in range(len(datasets)):
            dataset_path = os.path.join(output_dir, str(dataset_id))
            embeddings_list = []

            for batch_id in range(51):  # Assuming 50 embeddings per dataset
                file_path = os.path.join(dataset_path, f"{batch_id}.pt")
                if os.path.exists(file_path):
                    embedding = torch.load(file_path)  # Load tensor
                    embeddings_list.append(embedding.to(torch.float32).cpu().numpy())  # Convert to NumPy

            if embeddings_list:
                embeddings_dict[dataset_id] = np.vstack(embeddings_list)  # Stack arrays

        return embeddings_dict

    def plot_embeddings(embeddings_dict, output_dir):
        """Plot embeddings using t-SNE."""
        all_embeddings = np.concatenate(list(embeddings_dict.values()))
        dataset_labels = sum([[dataset_id] * len(embeddings) for dataset_id, embeddings in embeddings_dict.items()], [])

        perplexity = min(30, len(dataset_labels) - 1)  # Ensure perplexity is valid
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced = tsne.fit_transform(all_embeddings)

        plt.figure(figsize=(10, 6))
        for dataset_id in embeddings_dict.keys():
            dataset_name = list(config["datasets"])[dataset_id]
            indices = [i for i, label in enumerate(dataset_labels) if label == dataset_id]
            plt.scatter(reduced[indices, 0], reduced[indices, 1], label=dataset_name, alpha=0.7)

        plt.legend()
        plt.title("t-SNE Visualization of Dataset Embeddings")
        plt.savefig(f"{output_dir}/embeddings_plot.png")  # Save the plot as an image file
        plt.show()

    Logger.instance().set_log_dir(config["output_dir"])
    # Load base model once and share with Router to avoid double loading
    model, tokenizer = FastModel.from_pretrained(
        model_name=config["base_model"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        load_in_8bit=False,
        full_finetuning=False,
        dtype=torch.bfloat16,
    )
    router = Router(config, model, tokenizer)

    # Load the data
    Logger.instance().debug("Load data stream")
    data_stream = prepare_task_incremental_stream(config["data_stream"], config["datasets"],
                                                  router.tokenizer.eos_token, config)

    for dataset_id, dataset in enumerate(data_stream):
        save_dir = os.path.join(config["output_dir"], str(dataset_id))
        os.makedirs(save_dir, exist_ok=True)

        Logger.instance().debug(f"Extract embeddings for task {dataset_id}")
        trainloader, _, _ = init_dataloader(dataset, config['batch_size'])
        for batch_id, batch in enumerate(iter(trainloader)):
            embedding = router.extract_hidden_state(batch)
            torch.save(embedding, f"{save_dir}/{batch_id}.pt")

            if batch_id == 50:
                break

    # Load and plot embeddings
    embeddings = load_all_embeddings(config["datasets"], config["output_dir"])
    plot_embeddings(embeddings, config["output_dir"])


if __name__ == "__main__":
    config = {
        "seed": 0,
        "debug": False,
        "baseline": "callm",  # an instance as "callm" (CALLM), "inference_only", "train_single_lora"
        "base_model": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "load_in_4bit": True,
        "max_seq_length": 512,
        "data_stream": "TRACE-5000",
        "datasets": ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"],
        "evaluation_metrics": ["accuracy", "accuracy", "rouge", "edim", "accuracy", "accuracy", "accuracy", "sari"],
        "output_dir": os.path.expanduser("outputs/embeddings_TRACE5000"),
        "batch_size": 25,
        "num_epochs": 1,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "use_dora": False,
        "eval_step": 1,
        "save_step": 2,
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "optimizer": "adamw_8bit",
        "max_grad_norm": 0.3,
        "gradient_accumulation_steps": 8,
        "warmup_ratio": 0.01,
        "lr_scheduler_type": "linear",
        "checkpoint_path": None,
        "encoder_model": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "max_number_prototypes": 4,  # If None it depends on the available memory
        "topk": 1,
        "prompt_style": 'alpaca',
    }

    main(config)