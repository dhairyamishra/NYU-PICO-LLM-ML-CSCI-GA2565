import subprocess
import itertools
import os
import re
import glob
import gc
import torch
import argparse
from main import get_model_config  # ✅ Import from your main.py
import random
# Set a fixed seed for reproducibility
random.seed(71236)

# Define your hyperparameter grid
param_grid = {
    "--learning_rate": ["0.001", "0.005"],
    "--activation": ["relu", "gelu"],
    "--batch_size": ["32", "64", "128", "256"],
    "--embed_size": ["32", "64", "128", "256"],
    "--num_inner_mlp_layers": ["3", "5", "7", "9", "11"],
    "--kgram_k": ["1", "2", "3", "4", "5"],
    "--block_size": ["8", "16", "32", "64", "128"],
    "--num_epochs": ["2", "5", "7", "10"],
    "--tinystories_weight": ["0.8"],
    "--val_split": ["0.2"],

    # Fixed arguments
    "--train_subset_size": ["10000"],
    "--max_steps_per_epoch": ["10"],
    "--log_interval_steps": ["10"],
    "--sample_interval_seconds": ["60"],
    "--device_id": ["cuda:0"],  # ✅ must be valid for PyTorch
    "--prompt": ["Once upon a"],
    "--kgram_chunk_size": ["1", "2", "3"]  # required for model_config
}

# Create the logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Generate all combinations of parameters
# keys, values = zip(*param_grid.items())
keys, values = zip(*param_grid.items())
combinations = list(itertools.product(*values))
# possble values are 2^n, so we can limit the number of combinations
print(f"Total combinations: {len(combinations)}")
# Shuffle the seed so that quick testing has variey of data
# good mix of hyperparms can produce 7000+ permutations
random.shuffle(combinations)
combinations = combinations[:1]  # Limit to 100 combinations for quick testing
print(f"Random experiments to run: {len(combinations)}")

def strip_timestamp(model_config_str):
    return re.sub(r'_\d{8}_\d{6}$', '', model_config_str)
# Heuristic to determine model type based on provided arguments
def infer_model_name(args):
    if hasattr(args, "kgram_k") and hasattr(args, "num_inner_mlp_layers"):
        return "kgram_mlp_seq"
    elif hasattr(args, "rnn_hidden_dim"):
        return "lstm_seq"
    elif hasattr(args, "n_heads") and hasattr(args, "n_layers"):
        return "transformer"
    return "batch"  # Default to batch if no specific model type is found

def get_analyzed_configs_from_analysis_runs():
    analyzed = set()
    timestamp_pattern = r"_\d{8}_\d{6}$"

    # 1. From analysis_runs/*/generations/*
    for filepath in glob.glob("analysis_runs/*/generations/*"):
        base = os.path.basename(filepath)
        config = re.sub(timestamp_pattern, "", base)
        analyzed.add(config)

    # 2. From checkpoints/*
    if os.path.exists("checkpoints"):
        for folder in os.listdir("checkpoints"):
            if not os.path.isdir(os.path.join("checkpoints", folder)):
                continue
            config = re.sub(timestamp_pattern, "", folder)
            analyzed.add(config)

    return analyzed


# Load existing analysis results
analyzed_configs = get_analyzed_configs_from_analysis_runs()


def safe_filename(s):
    """Make string safe for Windows filenames."""
    s = s.replace(":", "-").replace(" ", "")
    return re.sub(r'[<>:"/\\|?*]', '', s)

num_runs_to_perform = 30
attempted_configs = set()

i = 0
while i < num_runs_to_perform and len(attempted_configs) < len(combinations):
    combo = random.choice(combinations)
    combo_key = tuple(combo)
    if combo_key in attempted_configs:
        continue
    attempted_configs.add(combo_key)

    args_dict = {key: val for key, val in zip(keys, combo)}
    args_namespace = argparse.Namespace(**{k.lstrip("--"): v for k, v in args_dict.items()})
    model_name = "batch"
    model_config_str = get_model_config(model_name, args_namespace)
    core_config = strip_timestamp(model_config_str)

    if core_config in analyzed_configs:
        print(f"⏩ Skipping already computed config: {core_config}")
        continue

    cmd = ["python", "main.py"]
    for key, val in args_dict.items():
        cmd.extend([key, val])

    safe_log_name = safe_filename(model_config_str)
    log_file = f"logs/{safe_log_name}.log"

    print(f"\n🔁 Running experiment {i+1}/{num_runs_to_perform}")
    print("Command:", " ".join(cmd))
    print("Log file:", log_file)

    try:
        with open(log_file, "w", encoding="utf-8") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT)
        torch.cuda.empty_cache()
        gc.collect()
        print(f"✅ Run {i+1} complete — log saved to {log_file}")
        i += 1
    except Exception as e:
        print(f"❌ Run {i+1} failed with error: {e} — see log file if available.")