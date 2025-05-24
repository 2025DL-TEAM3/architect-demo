import os
import random
from typing import Literal, Callable
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset, Sampler
import glob
import json
import os
import arc_utils
from copy import deepcopy
from datatypes import *


def build_hf_train_val_dataset(
    dataset_path: str,
    num_train_examples_per_normal_task: int = 3,
    num_datapoints_per_task: int = 50,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[HFDataset, HFDataset]:
    json_file_paths = sorted(glob.glob(f"{dataset_path}/*.json"))
    random.shuffle(json_file_paths)
    split = int(len(json_file_paths) * (1 - val_ratio))
    train_file_paths, val_file_paths = json_file_paths[:split], json_file_paths[split:]
    
    print(f"Train tasks: {len(train_file_paths)}, Validation tasks: {len(val_file_paths)}")
    
    train_tasks = arc_utils.load_tasks_from_paths(train_file_paths)
    val_tasks = arc_utils.load_tasks_from_paths(val_file_paths)
    
    train_datapoints = [
        arc_utils.sample_datapoints_from_normal_task(task, num_samples=num_train_examples_per_normal_task + 1)
        for task in train_tasks
        for _ in range(num_datapoints_per_task)
    ]
    
    val_datapoints = [
        arc_utils.sample_datapoints_from_normal_task(task, num_samples=num_train_examples_per_normal_task + 1)
        for task in val_tasks
        for _ in range(num_datapoints_per_task)
    ]
    
    train_challenges = dict()
    train_solutions = dict()
    for i, train_dp in enumerate(train_datapoints):
        task_id = train_dp["task_id"]
        train_ex = deepcopy(train_dp["train"])
        test_ex = deepcopy(train_dp["test"])
        test_sol = test_ex[0].pop("output")
        
        train_challenges[f"{task_id}-{i}"] = {
            "train": train_ex,
            "test": test_ex,
        }
        train_solutions[f"{task_id}-{i}"] = [test_sol]
    
    val_challenges = dict()
    val_solutions = dict()
    for i, val_dp in enumerate(val_datapoints):
        task_id = val_dp["task_id"]
        val_ex = deepcopy(val_dp["train"])
        test_ex = deepcopy(val_dp["test"])
        test_sol = test_ex[0].pop("output")
        
        val_challenges[f"{task_id}-{i}"] = {
            "train": val_ex,
            "test": test_ex,
        }
        val_solutions[f"{task_id}-{i}"] = [test_sol]
    
    print(len(train_challenges), len(train_solutions))
    print(len(val_challenges), len(val_solutions))  
    with open("input/my-arc/arc-agi_my_training_challenges.json", "w") as f:
        json.dump(train_challenges, f, indent=4)
    with open("input/my-arc/arc-agi_my_training_solutions.json", "w") as f:
        json.dump(train_solutions, f, indent=4)
    with open("input/my-arc/arc-agi_my_evaluation_challenges.json", "w") as f:
        json.dump(val_challenges, f, indent=4)
    with open("input/my-arc/arc-agi_my_evaluation_solutions.json", "w") as f:
        json.dump(val_solutions, f, indent=4)

    
    # Reset random seed
    random.seed()
    
build_hf_train_val_dataset(
    dataset_path="dataset"
)