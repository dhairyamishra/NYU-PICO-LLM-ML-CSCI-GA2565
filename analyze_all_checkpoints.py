import os
import re
import shutil
import argparse
import torch
import csv
from datetime import datetime
from analyze_checkpoints import analyze_checkpoints, plotlosses

def extract_config_from_dir(dirname):
    patterns = {
        "model_type": r"^(kgram_mlp_seq|lstm_seq|kvcache_transformer)",
        "inner_layers": r"mlp(\d+)",
        "k": r"k(\d+)",
        "chunk_size": r"cs(\d+)",
        "block_size": r"blk(\d+)",
        "embed_size": r"emb(\d+)",
        "activation": r"_act(relu|gelu)_",
        "learning_rate": r"lr([0-9.]+)",
        "batch_size": r"bs(\d+)",
        "tinystories_weight": r"tsw([0-9.]+)",
        "num_epochs": r"ep(\d+)",
    }

    config = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, dirname)
        if match:
            value = match.group(1)
            config[key] = int(value) if key not in {"activation", "model_type"} and '.' not in value else value
        else:
            config[key] = None
    return config

def safe_filename(s):
    return re.sub(r'[<>:"/\\|?*]', '', s.replace(" ", "_"))

def copy_analysis_outputs(src_dir, dst_base, model_type):
    dst_dir = os.path.join(dst_base, model_type, "generations")
    os.makedirs(dst_dir, exist_ok=True)

    for fname in ["generations.jsonl", "generations.csv", "metrics_curve.png"]:
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(dst_dir, os.path.basename(src_dir) + f"__{fname}")
            shutil.copyfile(src, dst)
            print(f"📄 Copied {fname} ➝ {dst}")

def already_analyzed(dst_base, model_type, dirname):
    dst_dir = os.path.join(dst_base, model_type, "generations")
    files = ["generations.jsonl", "metrics_curve.png"]
    return all(os.path.exists(os.path.join(dst_dir, dirname + f"__{f}")) for f in files)

def get_final_metrics(loss_log_path):
    if not os.path.exists(loss_log_path):
        return {}
    loss_dict = torch.load(loss_log_path, map_location="cpu")
    if not loss_dict:
        return {}
    last_epoch = max(int(k.split("_")[1]) for k in loss_dict.keys())
    metrics = loss_dict[f"epoch_{last_epoch}"]
    metrics["epoch"] = last_epoch
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", default="checkpoints", help="Top-level directory to search")
    parser.add_argument("--analysis_dir", default="analysis_runs", help="Where to copy analysis results")
    parser.add_argument("--prompt", default="Once upon a", help="Prompt for text generation")
    parser.add_argument("--skip_existing", action="store_true", help="Skip checkpoints already analyzed")
    args = parser.parse_args()

    checkpoint_root = args.checkpoint_root
    analysis_dir = args.analysis_dir
    all_dirs = [d for d in os.listdir(checkpoint_root) if os.path.isdir(os.path.join(checkpoint_root, d))]

    print(f"🧠 Found {len(all_dirs)} checkpoint runs to analyze.")
    summary_rows = []

    # Load summary cache
    summary_cache_path = os.path.join(analysis_dir, "summary_cache.csv")
    cached_configs = set()
    if os.path.exists(summary_cache_path):
        with open(summary_cache_path, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cached_configs.add(row["config_name"])

    for dirname in sorted(all_dirs):
        checkpoint_dir_sub = os.path.join(checkpoint_root, dirname)
        print(f"\n🔍 Analyzing: {dirname}")

        config = extract_config_from_dir(dirname)
        if None in config.values():
            print("🚫 Skipping due to incomplete config parsing.")
            continue

        if args.skip_existing:
            if dirname in cached_configs:
                print("✅ Skipping (already in summary cache).")
                continue
            if already_analyzed(args.analysis_dir, config["model_type"], dirname):
                print("✅ Skipping (output files exist).")
                continue

        # Wrap config into argparse.Namespace
        args_obj = argparse.Namespace(**config)
        setattr(args_obj, "checkpoint_dir_sub", checkpoint_dir_sub)
        setattr(args_obj, "prompt", args.prompt)

        # Run analysis
        analyze_checkpoints(
            checkpoint_dir_sub,
            config["model_type"],
            args.prompt,
            config["embed_size"],
            config["k"],
            config["chunk_size"],
            config["inner_layers"],
            config["block_size"],
            config["activation"]
        )

        loss_log_path = os.path.join(checkpoint_dir_sub, "loss_log.pt")
        plotlosses(loss_log_path, args_obj)

        # Copy results to analysis_runs
        copy_analysis_outputs(checkpoint_dir_sub, args.analysis_dir, config["model_type"])

        # Collect summary metrics
        final_metrics = get_final_metrics(loss_log_path)
        summary_row = {**config, **final_metrics, "config_name": dirname}
        summary_rows.append(summary_row)

    # Append to summary cache
    if summary_rows:
        keys = sorted(summary_rows[0].keys())
        os.makedirs(analysis_dir, exist_ok=True)
        write_header = not os.path.exists(summary_cache_path)
        with open(summary_cache_path, "a", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if write_header:
                writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n📊 Appended {len(summary_rows)} rows to summary cache.")
    else:
        print("No new summaries collected.")

if __name__ == "__main__":
    main()
