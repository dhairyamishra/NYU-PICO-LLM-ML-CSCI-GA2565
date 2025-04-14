import subprocess
import itertools
import os
import re
import argparse
from main import get_model_config  # ✅ Import from your main.py

# Define your hyperparameter grid
param_grid = {
    "--learning_rate": ["0.001"],
    # "--activation": ["relu", "gelu"],
    "--activation": ["gelu"],
    "--batch_size": ["16", "32", "64"],
    "--embed_size": ["64", "128", "256"],
    "--num_inner_mlp_layers": ["20"],
    "--kgram_k": ["2", "4"],
    "--block_size": ["128", "256"],
    "--num_epochs": ["5", "15"],
    "--tinystories_weight": ["0.8"],
    "--val_split": ["0.2"],

    # Fixed arguments
    "--train_subset_size": ["5000"],
    "--max_steps_per_epoch": ["30"],
    "--log_interval_steps": ["10"],
    "--sample_interval_seconds": ["30"],
    "--device_id": ["cuda:0"],  # ✅ must be valid for PyTorch
    "--prompt": ["Once upon a"],
    "--kgram_chunk_size": ["2"]  # required for model_config
}


easy_param_grid = {
    "--learning_rate": ["0.001"],
    "--activation": ["gelu"],
    "--batch_size": ["32"],
    "--embed_size": ["64"],
    "--num_inner_mlp_layers": ["20"],
    "--kgram_k": ["3"],
    "--block_size": ["128"],
    "--num_epochs": ["10"],
    "--tinystories_weight": ["0.8"],
    "--val_split": ["0.2"],

    # Fixed arguments
    "--train_subset_size": ["5000"],
    "--max_steps_per_epoch": ["30"],
    "--log_interval_steps": ["10"],
    "--sample_interval_seconds": ["30"],
    "--device_id": ["cuda:0"],
    "--prompt": ["Once upon a"],
    "--kgram_chunk_size": ["1"]
}
# Create the logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Generate all combinations of parameters
# keys, values = zip(*param_grid.items())
keys, values = zip(*easy_param_grid.items())
combinations = list(itertools.product(*values))

print(f"Total experiments to run: {len(combinations)}")

def safe_filename(s):
    """Make string safe for Windows filenames."""
    s = s.replace(":", "-").replace(" ", "")
    return re.sub(r'[<>:"/\\|?*]', '', s)

# Run each experiment
for i, combo in enumerate(combinations, 1):
    cmd = ["python", "main.py"]
    args_dict = {key: val for key, val in zip(keys, combo)}
    for key, val in args_dict.items():
        cmd.extend([key, val])

    # Convert args_dict to argparse.Namespace (like parse_args())
    args_namespace = argparse.Namespace(**{k.lstrip("--"): v for k, v in args_dict.items()})
    model_name = "batch"  # Neutral tag or could be "kgram_mlp_seq", etc.
    model_config_str = get_model_config(model_name, args_namespace)

    safe_log_name = safe_filename(model_config_str)
    log_file = f"logs/{safe_log_name}.log"

    print(f"\n🔁 Running experiment {i}/{len(combinations)}")
    print("Command:", " ".join(cmd))
    print("Log file:", log_file)

    with open(log_file, "w", encoding="utf-8") as logf:
        try:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            print(f"❌ Run {i} failed — see {log_file}")
        else:
            print(f"✅ Run {i} complete — log saved to {log_file}")
