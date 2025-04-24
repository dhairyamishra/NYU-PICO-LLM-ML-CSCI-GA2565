import os
import re
import csv
import json
import argparse
import shutil
from collections import OrderedDict

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import shap
import xgboost as xgb
from scipy.spatial import ConvexHull
import statsmodels.formula.api as smf

# === Directories ===
CHECKPOINT_ROOT = "checkpoints"
ANALYSIS_DIR = "analysis_runs"
PLOTS_DIR = os.path.join(ANALYSIS_DIR, "plots")
SUMMARY_PATH = os.path.join(ANALYSIS_DIR, "summary.csv")
os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# === Imports from your codebase ===
from analyze_checkpoints import analyze_checkpoints, plotlosses

def extract_config_from_dir(dirname):
    patterns = {
        "model_type": r"^(kgram_mlp_seq|lstm_seq|kvcache_transformer|deepseek_latent_attention)",
        "inner_layers": r"mlp(\d+)",
        "k": r"k(\d+)",
        "chunk_size": r"cs(\d+)",
        "block_size": r"blk(\d+)",
        "embed_size": r"emb(\d+)",
        "activation": r"_act(relu|gelu)(?:_|$)",
        "learning_rate": r"lr([0-9.]+)(?:_|$)",
        "batch_size": r"bs(\d+)(?:_|$)",
        "tinystories_weight": r"tsw([0-9.]+)(?:_|$)",
        "num_epochs": r"ep(\d+)(?:_|$)",
    }

    config = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, dirname)
        if match:
            value = match.group(1)
            config[key] = int(value) if key not in {"activation", "model_type"} and '.' not in value else value
        else:
            config[key] = None

    missing_keys = [k for k, v in config.items() if v is None]
    if missing_keys:
        print(f"⚠️ Skipping {dirname}: missing config keys {missing_keys}")
        return None

    return config

def run_checkpoint_analysis(prompt="Once upon a", skip_existing=True):
    summary_rows = []
    summary_cache_path = os.path.join(ANALYSIS_DIR, "summary_cache.csv")
    all_dirs = [d for d in os.listdir(CHECKPOINT_ROOT) if os.path.isdir(os.path.join(CHECKPOINT_ROOT, d))]
    cached = set()

    if os.path.exists(summary_cache_path):
        with open(summary_cache_path, newline='', encoding="utf-8") as f:
            cached = set(row["config_name"] for row in csv.DictReader(f))

    for dirname in sorted(all_dirs):
        if skip_existing and dirname in cached:
            continue

        config = extract_config_from_dir(dirname)
        if config is None:
            continue

        checkpoint_dir = os.path.join(CHECKPOINT_ROOT, dirname)
        args_obj = argparse.Namespace(**config)
        setattr(args_obj, "checkpoint_dir_sub", checkpoint_dir)
        setattr(args_obj, "prompt", prompt)

        try:
            analyze_checkpoints(
                checkpoint_dir, config["model_type"], prompt,
                config["embed_size"], config["k"], config["chunk_size"],
                config["inner_layers"], config["block_size"], config["activation"]
            )
            plotlosses(checkpoint_dir, args_obj)

            loss_path = os.path.join(checkpoint_dir, "loss_log.pt")
            loss_dict = torch.load(loss_path, map_location="cpu")
            last_epoch = max(int(k.split("_")[1]) for k in loss_dict.keys())
            metrics = loss_dict[f"epoch_{last_epoch}"]
            metrics["epoch"] = last_epoch

            summary_rows.append({**config, **metrics, "config_name": dirname})

        except Exception as e:
            print(f"❌ Failed to analyze {dirname}: {e}")

    # ✅ Write summary_cache.csv
    if summary_rows:
        keys = sorted(summary_rows[0].keys())
        with open(summary_cache_path, "a", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if os.stat(summary_cache_path).st_size == 0:
                writer.writeheader()
            writer.writerows(summary_rows)
        print(f"✅ Appended {len(summary_rows)} rows to summary cache.")
    else:
        print("⚠️ No valid checkpoint runs were added to summary cache.")

def full_summary_analysis():
    os.makedirs(PLOTS_DIR, exist_ok=True)  # ✅ Ensure plot dir exists

    if not os.path.exists(SUMMARY_PATH):
        print("❌ summary.csv not found. Run with --run_checkpoints first or make sure summary_cache.csv exists.")
        return

    df = pd.read_csv(SUMMARY_PATH)
    numeric_cols = ["avg_loss", "val_loss", "perplexity", "token_accuracy", "learning_rate",
                    "embed_size", "k", "chunk_size", "inner_layers", "block_size", "batch_size", "tinystories_weight"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=["val_loss", "token_accuracy", "model_type"])

    # Pareto Frontier
    points = df[["val_loss", "token_accuracy"]].dropna().values
    if len(points) >= 3:
        hull = ConvexHull(points)
        pareto_points = points[hull.vertices][np.argsort(points[hull.vertices][:, 0])]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x="token_accuracy", y="val_loss", hue="model_type", style="activation", ax=ax)
        ax.plot(pareto_points[:, 1], pareto_points[:, 0], "r--", label="Pareto Frontier")
        ax.set_title("Pareto Frontier: Accuracy vs Validation Loss")
        fig.tight_layout()
        fig.savefig(f"{PLOTS_DIR}/pareto_frontier_overlay.png")
        plt.close(fig)

    # SHAP Analysis
    df = df.loc[:, ~df.columns.duplicated()]
    X = df[numeric_cols].fillna(0).drop(columns=["val_loss"])
    y = df["val_loss"].fillna(0)
    model = xgb.XGBRegressor(tree_method="hist")
    model.fit(X, y)
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP Hyperparameter Importance")
    plt.savefig(f"{PLOTS_DIR}/shap_summary.png", bbox_inches='tight')
    plt.close()

    # Save Top Configs
    top10 = df.sort_values("val_loss").head(10)
    top10.to_csv(f"{PLOTS_DIR}/top10_val_loss.csv", index=False)

    print("✅ Summary analysis complete. All plots saved in:", PLOTS_DIR)

def merge_summary_cache():
    merged = OrderedDict()
    for fname in os.listdir(ANALYSIS_DIR):
        if fname.startswith("summary_cache") and fname.endswith(".csv"):
            with open(os.path.join(ANALYSIS_DIR, fname), newline='', encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    merged[row["config_name"]] = row
    if merged:
        with open(SUMMARY_PATH, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(merged.values())[0].keys())
            writer.writeheader()
            writer.writerows(merged.values())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_checkpoints", action="store_true", help="Run per-checkpoint analysis")
    parser.add_argument("--skip_existing", action="store_true", help="Skip checkpoints already analyzed")
    parser.add_argument("--prompt", default="Once upon a", help="Prompt for generation")
    args = parser.parse_args()

    if args.run_checkpoints:
        run_checkpoint_analysis(prompt=args.prompt, skip_existing=args.skip_existing)

    merge_summary_cache()
    full_summary_analysis()

if __name__ == "__main__":
    main()
