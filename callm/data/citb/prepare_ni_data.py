""" Modified from https://github.com/hyintell/CITB/tree/main """

import os
import json
import random
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from collections import defaultdict
from callm.utils import Logger
from callm.prompts_repo import PromptBuilder


# CITB data stream metric mappings
CITB_METRICS = {
    "CITB": "rouge",    # All tasks in CITB use rouge
    "CITB-38": "rouge"  # All tasks in CITB-38 use rouge
}


def get_citb_metric_for_stream(data_stream: str) -> str:
    """Get the ground truth metric for CITB data streams.
    
    Args:
        data_stream: The CITB data stream name ('CITB' or 'CITB-38')
        
    Returns:
        The metric name for the stream, or None if not found
    """
    return CITB_METRICS.get(data_stream)

def build_task_datasets(task_instances):
    """Convert dict[task_name] -> list[instances] into dict[task_name] -> Dataset(prompt, answer)."""
    task_datasets = {}
    for task_name, instances in task_instances.items():
        prompts, answers = [], []
        for ex in instances:
            definition = ex.get("Definition", [""])[0]
            inp = ex["Instance"]["input"]
            out = ex["Instance"]["output"][0] if ex["Instance"]["output"] else ""

            prompt = definition.strip() + "\n" + inp.strip()
            answer = out.strip()

            prompts.append(prompt)
            answers.append(answer)

        task_datasets[task_name] = Dataset.from_dict({"prompt": prompts, "answer": answers})
    return task_datasets


def get_task2instance(dataset):
    """ Return a new dict, key: task name, value: list of instances belong to the task
    dataset: list of dicts, each dict is an instance
    """

    # task to list of instances mapping
    task2instance = defaultdict(list)
    for instance in dataset:
        instance_task = instance["Task"]
        task2instance[instance_task].append(instance)

    task2instances_len = {task_name: len(instances) for task_name, instances in task2instance.items()}
    print(f"task2instances_len: {task2instances_len}")

    return task2instance


def train_dev_test_split_by_task(raw_datasets, max_num_instances_per_task, max_num_instances_per_eval_task, continual=False):
    """
    For each task, do train/test split.
    The number of test instances is set by max_num_instances_per_eval_task.
    We set the number of dev instances to zero.
    """

    # get all task names 
    all_task_names = set(i["Task"] for i in raw_datasets['train'])
    print(f"all_task_names: {len(all_task_names)}")

    task2instance = get_task2instance(raw_datasets['train'])

    train_instances, test_instances = defaultdict(list), defaultdict(list)

    # split each tasks' instances into train and test
    for task_name, instances in task2instance.items():
        test_instances[task_name].extend(instances[:max_num_instances_per_eval_task])

        # make sure per task training instances not exceeding the limit
        remaining_instances = instances[max_num_instances_per_eval_task:]
        print(f"total: {len(instances)}, remaining_instances: {len(remaining_instances)}")
        if len(remaining_instances) >= max_num_instances_per_task:
            random.shuffle(remaining_instances)
            train_instances[task_name].extend(remaining_instances[:max_num_instances_per_task])
        else:
            train_instances[task_name].extend(remaining_instances)

        Logger.instance().debug(f'Task name: {task_name}. Length instances: {len(instances)}')
        Logger.instance().debug(f'Task name: {task_name}. Length train instances: {len(train_instances[task_name])}')
        Logger.instance().debug(f'Task name: {task_name}. Length test instances: {len(test_instances[task_name])}')

    print(f"test_instances tasks: {len(test_instances.keys())}")
    print(f"train_instances tasks: {len(train_instances.keys())}")

    return train_instances, test_instances


def prepare_single_dataset_alpaca_format(data, eos_token: str, debug: bool, prompt_style: str = "alpaca"):
    eos = eos_token or ""

    def formatting_prompts_func(examples):
        texts = []
        for instruction, output in zip(examples["prompt"], examples["answer"]):
            text = PromptBuilder.build_generation_prompt(
                instruction=str(instruction),
                style=prompt_style,
                response=str(output),
            ) + eos
            texts.append(text)
        return {"text": texts}

    dataset = data.map(formatting_prompts_func, batched=True)
    return dataset


def build_per_task_datasetdicts(train_instances, test_instances, eos_token, debug=False, prompt_style="alpaca"):
    """Return dict[task_name] -> DatasetDict(train/dev/test)."""

    train_datasets = build_task_datasets(train_instances)
    test_datasets  = build_task_datasets(test_instances)

    per_task_datasetdicts = {}

    all_task_names = set(train_datasets.keys()) | set(test_datasets.keys())
    for task_name in all_task_names:
        train_ds = train_datasets.get(task_name, Dataset.from_dict({"prompt":[],"answer":[]}))
        test_ds  = test_datasets.get(task_name, Dataset.from_dict({"prompt":[],"answer":[]}))

        # NOTE: CITB naturally provides train/test only, but we need train/eval/test.
        # Split test data: 50% for eval, 50% for test to avoid data leakage.
        formatted_train = prepare_single_dataset_alpaca_format(train_ds, eos_token, debug, prompt_style)
        formatted_test  = prepare_single_dataset_alpaca_format(test_ds, eos_token, debug, prompt_style)
        
        # Split test data in half: first half as eval, second half as test
        test_size = len(formatted_test)
        if test_size > 1:
            eval_indices = list(range(test_size // 2))
            test_indices = list(range(test_size // 2, test_size))
            
            eval_split = formatted_test.select(eval_indices)
            test_split = formatted_test.select(test_indices)
        else:
            # If only 1 test example, use it for both (edge case)
            eval_split = formatted_test
            test_split = formatted_test

        per_task_datasetdicts[task_name] = DatasetDict({
            "train": formatted_train,
            "eval":  eval_split,
            "test":  test_split,
        })

    return per_task_datasetdicts


def prepare_stream_citb(data_stream: str, config_path: str, eos_token: str, synthetic_data_path=None, debug=False, prompt_style="alpaca"):

    if debug:
        Logger.instance().warning("No debug version for CITB dataset. Use TRACE dataset.")
    
    # Load JSON file
    with open(config_path, "r") as file:
        data_config = json.load(file)

    if data_stream == "CITB":
        Logger.instance().debug("Training with CITB")
        data_config["task_split_file_name"] = "cl_dialogue_tasks"
    elif data_stream == "CITB-38":
        Logger.instance().debug("Training with CITB-38")
        data_config["task_split_file_name"] = "cl_38_random_tasks"

    # Load task list from split file
    split_file_path = os.path.join(data_config["data_dir"], f"{data_config['task_split_file_name']}.txt")
    with open(split_file_path, "r") as f:
        task_names = [line.strip() for line in f if line.strip()]
    
    Logger.instance().debug(f"Loading {len(task_names)} tasks from {split_file_path}")
    
    # Load tasks directly from JSON files instead of using dataset script
    all_examples = []
    for task_name in task_names:
        task_file_path = os.path.join(data_config["task_dir"], f"{task_name}.json")
        with open(task_file_path, "r") as f:
            task_data = json.load(f)
        
        # Extract instances and add task metadata
        instances = task_data.get("Instances", [])
        for instance in instances:
            example = {
                "Task": task_name,
                "id": instance["id"],
                "Instance": instance,
                "Definition": task_data.get("Definition", []),
                "Positive Examples": task_data.get("Positive Examples", []),
                "Negative Examples": task_data.get("Negative Examples", [])
            }
            all_examples.append(example)
    
    # Create dataset from examples
    from datasets import Dataset
    raw_datasets = {"train": Dataset.from_list(all_examples)}
    
    train_instances, test_instances = train_dev_test_split_by_task(raw_datasets,
        max_num_instances_per_task=data_config["max_num_instances_per_task"],
        max_num_instances_per_eval_task=data_config["max_num_instances_per_eval_task"],
        continual=True
    )

    per_task_datasetdicts = build_per_task_datasetdicts(
        train_instances, test_instances,
        eos_token=eos_token, debug=debug, prompt_style=prompt_style
        )

    stream = []
    for task_name, task_data in per_task_datasetdicts.items():
        if synthetic_data_path:
            Logger.instance().debug("Loading and appending synthetic data to train sets.")

            original_train_set = task_data["train"]
            synthetic_filename = os.path.join(synthetic_data_path, f"{task_name}.json")
            if os.path.exists(synthetic_filename):
                Logger.instance().debug(f"Task {task_name}: Found synthetic data at {synthetic_filename}")
                synth_data = load_dataset("json", data_files=synthetic_filename, split="train")
                combined_train_set = concatenate_datasets([original_train_set, synth_data])
                task_data['train'] = combined_train_set

                Logger.instance().debug(
                    f"Task {task_name}: Added {len(synth_data)} synthetic examples. "
                    f"New train size: {len(combined_train_set)}"
                )
            else:
                Logger.instance().debug(f"Task {task_name}: No synthetic data found at {synthetic_filename}")

        stream.append(task_data)
    return stream, per_task_datasetdicts.keys()
