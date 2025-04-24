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

    # Lineplot: Val Loss vs Embed Size by Activation
    # Heatmaps: Val Loss vs (embed_size, k) per model_type
    # Box + Strip by Activation 
    # Regression: Residuals vs Predicted
    add_extra_plots(df)

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

def add_extra_plots(df):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    palette = "Set2"
    model_markers = ["o", "s", "D", "^", "P", "X"]

    numeric_cols = [
        'avg_loss', 'val_loss', 'perplexity', 'token_accuracy',
        'learning_rate', 'embed_size', 'k', 'chunk_size', 'inner_layers',
        'block_size', 'batch_size', 'tinystories_weight'
    ]

    # === Correlation Matrix ===
    if len(df[numeric_cols].dropna()) >= 2:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Matrix (Hyperparams vs Metrics)")
        fig.tight_layout()
        fig.savefig(f"{PLOTS_DIR}/correlation_matrix.png", bbox_inches='tight')
        plt.close(fig)

    # === Regression: Residuals vs Predicted ===
    reg_df = df.dropna(subset=["val_loss", "embed_size", "activation"]).copy()
    if len(reg_df) >= 5:
        model = smf.ols("val_loss ~ embed_size + C(activation)", data=reg_df).fit()
        reg_df["predicted_val_loss"] = model.predict(reg_df)
        reg_df["residual"] = reg_df["val_loss"] - reg_df["predicted_val_loss"]

        with open(os.path.join(PLOTS_DIR, "regression_summary.txt"), "w") as f:
            f.write(str(model.summary()))

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=reg_df,
            x="predicted_val_loss", y="residual",
            hue="model_type", style="activation", palette=palette, s=80, ax=ax
        )
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_title("Residuals vs Predicted Validation Loss")
        fig.tight_layout()
        fig.savefig(f"{PLOTS_DIR}/regression_residuals.png", bbox_inches='tight')
        plt.close(fig)

    # === Box + Strip by Activation ===
    if "activation" in df.columns and "val_loss" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        order = df.groupby("activation")["val_loss"].median().sort_values().index
        sns.boxplot(data=df, x='activation', y='val_loss', hue='model_type', order=order, palette=palette, ax=ax)
        sns.stripplot(data=df, x='activation', y='val_loss', hue='model_type', dodge=True, alpha=0.5, order=order, palette=palette, ax=ax)
        ax.set_title("Validation Loss by Activation Function (Ordered by Median)")
        ax.grid(True)
        fig.tight_layout()
        ax.legend(title="Model Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.savefig(f"{PLOTS_DIR}/val_loss_by_activation_ordered.png", bbox_inches='tight')
        plt.close(fig)

    # === Lineplot: Val Loss vs Embed Size by Activation ===
    if "embed_size" in df.columns and "val_loss" in df.columns:
        g = sns.lmplot(
            data=df, x='embed_size', y='val_loss',
            hue='model_type', col='activation',
            palette=palette,
            markers=model_markers[:df["model_type"].nunique()],
            height=5, aspect=1.2, ci=None,
            scatter_kws={'s': 60},
            legend=False
        )
        g.set_titles(col_template="{col_name} activation")
        g.fig.subplots_adjust(top=0.85, right=0.72)
        g.fig.suptitle("Validation Loss vs Embedding Size by Activation", fontsize=16)

        handles, labels = g.axes[0][0].get_legend_handles_labels()
        g.fig.legend(
            handles, labels,
            title="Model Type",
            loc='center right',
            bbox_to_anchor=(1, 0.5),
            borderaxespad=0.
        )

        g.savefig(f"{PLOTS_DIR}/val_loss_vs_embed_size_by_activation.png", bbox_inches='tight')
        plt.close(g.fig)

    # === Heatmaps: Val Loss vs (embed_size, k) per model_type ===
    if {"embed_size", "k", "val_loss", "model_type"}.issubset(df.columns):
        heatmap_df = df[["embed_size", "k", "val_loss", "model_type"]].dropna()
        if not heatmap_df.empty:
            for model in heatmap_df["model_type"].unique():
                model_df = heatmap_df[heatmap_df["model_type"] == model]
                pivot = model_df.pivot_table(index="embed_size", columns="k", values="val_loss", aggfunc="mean")
                if pivot.empty:
                    continue
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis",
                            linewidths=0.5, cbar_kws={"label": "Validation Loss"}, ax=ax)
                ax.set_title(f"Validation Loss Heatmap: Embed Size vs k ({model})")
                ax.set_xlabel("k")
                ax.set_ylabel("Embedding Size")
                fig.tight_layout()
                fig.savefig(f"{PLOTS_DIR}/val_loss_heatmap_{model}_embed_k.png", bbox_inches="tight")
                plt.close(fig)



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
