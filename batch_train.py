import subprocess
import itertools
import os
import re
import glob
import gc
import torch
import argparse
import random
from main import get_model_config  # ✅ Import from your main.py

random.seed(0)

# Define your hyperparameter grid
param_grid = {
    # "--model_type": ["deepseek_latent_attention"],  # ✅ Use only one model type for simplicity
    "--model_type": ["deepseek_latent_attention", "lstm_seq", "kvcache_transformer", "kgram_mlp_seq"],  # ✅ Added model type
    "--learning_rate": ["0.001",],
    "--activation": [ "gelu"],
    "--batch_size": ["32"],
    "--embed_size": ["32"],
    "--num_inner_mlp_layers": ["3", "5"],
    "--kgram_k": ["2", "3"],
    "--kgram_chunk_size": ["1", "2"],
    "--block_size": ["64"],
    "--num_epochs": ["3"],
    "--tinystories_weight": ["0.9"],
    "--val_split": ["0.2"],
    "--train_subset_size": ["5000"],
    "--max_steps_per_epoch": ["5"],
    "--log_interval_steps": ["10"],
    "--sample_interval_seconds": ["60"],
    "--device_id": ["cuda:0"],
    "--prompt": ["Once upon a"]
}

os.makedirs("logs", exist_ok=True)

keys, values = zip(*param_grid.items())
combinations = list(itertools.product(*values))
print(f"Total combinations: {len(combinations)}")
# Filter out combinations with empty values
random.shuffle(combinations)
combinations = combinations[:20]  # ✅ Trim runs

def strip_timestamp(model_config_str):
    return re.sub(r'_\d{8}_\d{6}$', '', model_config_str)

def get_analyzed_configs_from_analysis_runs():
    analyzed = set()
    ts_pattern = r"_\d{8}_\d{6}$"
    for folder in glob.glob("checkpoints/*"):
        if os.path.isdir(folder):
            base = os.path.basename(folder)
            analyzed.add(re.sub(ts_pattern, "", base))
    return analyzed

analyzed_configs = get_analyzed_configs_from_analysis_runs()

def safe_filename(s):
    return re.sub(r'[<>:"/\\|?*]', '', s.replace(":", "-").replace(" ", ""))

i = 0
attempted_configs = set()
num_runs_to_perform = 10

while i < num_runs_to_perform and len(attempted_configs) < len(combinations):
    combo = random.choice(combinations)
    combo_key = tuple(combo)
    if combo_key in attempted_configs:
        continue
    attempted_configs.add(combo_key)

    args_dict = {key: val for key, val in zip(keys, combo)}
    args_namespace = argparse.Namespace(**{k.lstrip("--"): v for k, v in args_dict.items()})

    model_name = args_namespace.model_type
    model_config_str = get_model_config(model_name, args_namespace)
    core_config = strip_timestamp(model_config_str)

    if core_config in analyzed_configs:
        print(f"⏩ Skipping already computed config: {core_config}")
        continue

    cmd = ["python", "main.py"]
    for key, val in args_dict.items():
        cmd.extend([key, val])

    log_file = f"logs/{safe_filename(model_config_str)}.log"

    print(f"\n🔁 Running experiment {i + 1}/{num_runs_to_perform}")
    print("Command:", " ".join(cmd))
    print("Log file:", log_file)

    try:
        with open(log_file, "w", encoding="utf-8") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT)
        torch.cuda.empty_cache()
        gc.collect()
        print(f"✅ Run {i + 1} complete — log saved to {log_file}")
        i += 1
    except Exception as e:
        print(f"❌ Run {i + 1} failed with error: {e}")