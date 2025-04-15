import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from collections import OrderedDict
from scipy.spatial import ConvexHull
import numpy as np
import statsmodels.formula.api as smf

# === Config ===
analysis_dir = "analysis_runs"
summary_path = os.path.join(analysis_dir, "summary.csv")
output_dir = os.path.join(analysis_dir, "plots")
os.makedirs(output_dir, exist_ok=True)

# === Merge Summaries ===
def merge_summaries():
    merged_rows = OrderedDict()
    for fname in os.listdir(analysis_dir):
        if fname.startswith("summary_cache") and fname.endswith(".csv"):
            print(f"📥 Reading {fname}")
            full_path = os.path.join(analysis_dir, fname)
            with open(full_path, newline='', encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    config_name = row["config_name"]
                    merged_rows[config_name] = row

    if not merged_rows:
        print("⚠️ No summary_cache files found.")
        return

    output_path = summary_path
    keys = list(next(iter(merged_rows.values())).keys())
    with open(output_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(merged_rows.values())
    print(f"✅ Merged summary written to {output_path} ({len(merged_rows)} entries)")

# === Load and Clean ===
merge_summaries()
if not os.path.exists(summary_path):
    raise FileNotFoundError("❌ summary.csv not found. Run analyze_all_checkpoints.py first.")

df = pd.read_csv(summary_path)
numeric_cols = [
    'avg_loss', 'val_loss', 'perplexity', 'token_accuracy',
    'learning_rate', 'embed_size', 'k', 'chunk_size', 'inner_layers',
    'block_size', 'batch_size', 'tinystories_weight'
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')


# === Compute Top 10 Leaderboard (in memory) ===
top10 = df.sort_values("val_loss").head(10).copy()
top10["rank"] = range(1, len(top10) + 1)  # ✅ Dynamically assign correct number of ranks
top10_labels = dict(zip(top10["config_name"], top10["rank"]))

# === Save Leaderboard ===
top10_path = f"{output_dir}/top10_val_loss.csv"
top10.to_csv(top10_path, index=False)
print(f"✅ Top 10 configs saved to: {top10_path}")

# === Plot Settings ===
palette = "Set2"
model_markers = ["o", "s", "D", "^", "P", "X"]

# === Plot Helper: Draw top10 marker labels ===
def annotate_top10(ax, df, x_col, y_col):
    for _, row in df.iterrows():
        if row["config_name"] in top10_labels:
            rank = top10_labels[row["config_name"]]
            ax.text(row[x_col], row[y_col], str(rank), fontsize=8, fontweight='bold', ha='center', va='center', color='black',
                    bbox=dict(boxstyle="circle,pad=0.2", fc="yellow", ec="black", lw=0.5))

def add_top10_legend(fig):
    legend_text = "\n".join(
        f"{r}. {cfg}" for r, cfg in zip(top10["rank"], top10["config_name"])
    )
    fig.text(1.02, 0.5, legend_text, fontsize=7, va='center', ha='left')
# === Plot 1: Pareto Frontier Overlay ===
subset = df.dropna(subset=["val_loss", "token_accuracy"])
points = subset[["val_loss", "token_accuracy"]].values

if len(points) >= 3:
    try:
        hull = ConvexHull(points)
        pareto_points = points[hull.vertices]
        pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=subset, x="token_accuracy", y="val_loss",
            hue="model_type", style="activation", palette=palette, s=80, ax=ax
        )
        ax.plot(pareto_points[:, 1], pareto_points[:, 0], "r--", label="Pareto Frontier")
        annotate_top10(ax, subset, "token_accuracy", "val_loss")
        add_top10_legend(fig)

        ax.set_title("Pareto Frontier: Token Accuracy vs Validation Loss")
        ax.set_xlabel("Token Accuracy")
        ax.set_ylabel("Validation Loss")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/pareto_frontier_overlay.png", bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"⚠️ Could not compute Pareto frontier: {e}")

# === Plot 2: Token Accuracy vs Validation Loss (custom scatter plot) ===
if len(subset) >= 3:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=subset,
        x="token_accuracy", y="val_loss",
        hue="model_type", style="activation",
        palette=palette, s=70, ax=ax
    )
    annotate_top10(ax, subset, "token_accuracy", "val_loss")
    add_top10_legend(fig)

    ax.set_title("Token Accuracy vs Validation Loss (Custom Plot)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/jointplot_tokenacc_valloss.png", bbox_inches='tight')
    plt.close(fig)

# === Plot 3: Regression + Residuals ===
reg_df = df.dropna(subset=["val_loss", "embed_size", "activation"]).copy()
if len(reg_df) >= 5:
    model = smf.ols("val_loss ~ embed_size + C(activation)", data=reg_df).fit()
    reg_df["predicted_val_loss"] = model.predict(reg_df)
    reg_df["residual"] = reg_df["val_loss"] - reg_df["predicted_val_loss"]

    with open(os.path.join(output_dir, "regression_summary.txt"), "w") as f:
        f.write(str(model.summary()))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=reg_df,
        x="predicted_val_loss", y="residual",
        hue="model_type", style="activation", palette=palette, s=80, ax=ax
    )
    ax.axhline(0, color="gray", linestyle="--")
    annotate_top10(ax, reg_df, "predicted_val_loss", "residual")
    add_top10_legend(fig)

    ax.set_title("Residuals vs Predicted Validation Loss")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/regression_residuals.png", bbox_inches='tight')
    plt.close(fig)

# === Plot 4: Val Loss vs Embed Size by Activation ===
if "embed_size" in df.columns and "val_loss" in df.columns:
    g = sns.lmplot(
        data=df, x='embed_size', y='val_loss',
        hue='model_type', col='activation',
        palette=palette,
        markers=model_markers[:df["model_type"].nunique()],
        height=5, aspect=1.2, ci=None,
        scatter_kws={'s': 60},
        legend=False  # ✅ Prevent auto subplot legends
    )
    g.set_titles(col_template="{col_name} activation")
    g.fig.subplots_adjust(top=0.85, right=0.72)  # ✅ More right space
    g.fig.suptitle("Validation Loss vs Embedding Size by Activation", fontsize=16)

    # === Annotate top-10 within each subplot
    for ax in g.axes.flatten():
        activation_title = ax.get_title()
        activation_name = activation_title.split()[0]
        subset_ax = df[df["activation"] == activation_name]
        annotate_top10(ax, subset_ax, "embed_size", "val_loss")

    # === Add shared seaborn legend manually
    handles, labels = g.axes[0][0].get_legend_handles_labels()
    g.fig.legend(
        handles, labels,
        title="Model Type",
        loc='center right',
        bbox_to_anchor=(1, 0.5),
        borderaxespad=0.
    )

    # === Add custom top-10 legend
    legend_text = "\n".join(
        f"{r}. {cfg}" for r, cfg in zip(top10["rank"], top10["config_name"])
    )
    g.fig.text(1.05, 0.5, legend_text, fontsize=7, va='center', ha='left')

    g.savefig(f"{output_dir}/val_loss_vs_embed_size_by_activation.png", bbox_inches='tight')
    plt.close(g.fig)

# === Plot 5: Perplexity vs Accuracy ===
if "perplexity" in df.columns and "token_accuracy" in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='token_accuracy', y='perplexity',
        hue='model_type', style='activation',
        palette=palette, s=100, ax=ax
    )
    ax.set_yscale('log')
    ax.set_title("Perplexity vs Token Accuracy (log scale)")
    ax.grid(True)
    annotate_top10(ax, df, "token_accuracy", "perplexity")
    add_top10_legend(fig)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/perplexity_vs_accuracy_log.png", bbox_inches='tight')
    plt.close(fig)

# === Plot 6: Box + Strip by Activation ===
if "activation" in df.columns and "val_loss" in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    order = df.groupby("activation")["val_loss"].median().sort_values().index
    sns.boxplot(data=df, x='activation', y='val_loss', hue='model_type', order=order, palette=palette, ax=ax)
    sns.stripplot(data=df, x='activation', y='val_loss', hue='model_type', dodge=True, alpha=0.5, order=order, palette=palette, ax=ax)
    annotate_top10(ax, df, "activation", "val_loss")
    add_top10_legend(fig)
    ax.set_title("Validation Loss by Activation Function (Ordered by Median)")
    ax.grid(True)
    fig.tight_layout()
    ax.legend(title="Model Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.savefig(f"{output_dir}/val_loss_by_activation_ordered.png", bbox_inches='tight')
    plt.close(fig)

# === Plot 8: Correlation Matrix ===
if len(df[numeric_cols].dropna()) >= 2:
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    add_top10_legend(fig)
    ax.set_title("Correlation Matrix (Hyperparams vs Metrics)")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/correlation_matrix.png", bbox_inches='tight')
    plt.close(fig)

# === Plot 9: Heatmap of val_loss vs (embed_size, k), per model_type ===
if {"embed_size", "k", "val_loss", "model_type"}.issubset(df.columns):
    heatmap_df = df[["embed_size", "k", "val_loss", "model_type"]].dropna()
    if heatmap_df.empty:
        print("⚠️ Not enough data for embed_size vs k heatmap")
    else:
        unique_models = heatmap_df["model_type"].unique()
        for model in unique_models:
            model_df = heatmap_df[heatmap_df["model_type"] == model]
            if model_df.empty:
                print(f"⚠️ Skipping heatmap: No data for model_type={model}")
                continue

            pivot = model_df.pivot_table(
                index="embed_size", columns="k", values="val_loss", aggfunc="mean"
            )

            if pivot.empty:
                print(f"⚠️ Skipping heatmap: Not enough data for {model}")
                continue

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".3f",
                cmap="viridis",
                linewidths=0.5,
                cbar_kws={"label": "Validation Loss"},
                ax=ax
            )
            ax.set_title(f"Validation Loss Heatmap: Embed Size vs k ({model})")
            ax.set_xlabel("k")
            ax.set_ylabel("Embedding Size")
            fig.tight_layout()
            heatmap_path = f"{output_dir}/val_loss_heatmap_{model}_embed_k.png"
            fig.savefig(heatmap_path, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ Saved heatmap: {heatmap_path}")


print("✅ All analysis complete.")