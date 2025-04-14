import subprocess
import itertools
import os
import re
import glob
import argparse
from main import get_model_config  # ✅ Import from your main.py
import random
# Set a fixed seed for reproducibility
random.seed(42)

# Define your hyperparameter grid
param_grid = {
    "--learning_rate": ["0.001", "0.05"],
    "--activation": ["relu", "gelu"],
    "--batch_size": ["4", "8", "16"],
    "--embed_size": ["16", "32", "64"],
    "--num_inner_mlp_layers": ["3", "10"],
    "--kgram_k": ["1", "2", "3", "4"],
    "--block_size": ["8", "16", "32"],
    "--num_epochs": ["2", "5", "7"],
    "--tinystories_weight": ["0.8"],
    "--val_split": ["0.2"],

    # Fixed arguments
    "--train_subset_size": ["1000"],
    "--max_steps_per_epoch": ["20"],
    "--log_interval_steps": ["19"],
    "--sample_interval_seconds": ["30"],
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
# Shuffle the seed so that quick testing has variey of data
# good mix of hyperparms can produce 7000+ permutations
random.shuffle(combinations)
combinations = combinations[:15]  # Limit to 100 combinations for quick testing
print(f"Total experiments to run: {len(combinations)}")

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

# Run each experiment
for i, combo in enumerate(combinations, 1):
    try:
        cmd = ["python", "main.py"]
        args_dict = {key: val for key, val in zip(keys, combo)}
        for key, val in args_dict.items():
            cmd.extend([key, val])

        args_namespace = argparse.Namespace(**{k.lstrip("--"): v for k, v in args_dict.items()})
        # model_name = infer_model_name(args_namespace)
        model_name = "batch"
        model_config_str = get_model_config(model_name, args_namespace)
        core_config = strip_timestamp(model_config_str)
        safe_log_name = safe_filename(model_config_str)
        log_file = f"logs/{safe_log_name}.log"

        print(f"\n🔁 Running experiment {i}/{len(combinations)}")
        print("Command:", " ".join(cmd))
        print("Log file:", log_file)

        if core_config in analyzed_configs:
            print(f"⏩ Skipping already computed config: {core_config}")
            continue

        with open(log_file, "w", encoding="utf-8") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT)

        print(f"✅ Run {i} complete — log saved to {log_file}")

    except Exception as e:
        print(f"❌ Run {i} failed with error: {e} — see log file if available.")
