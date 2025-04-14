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
    print(f"\n✅ Merged summary written to {output_path} ({len(merged_rows)} entries)")


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


# === Plot 1: Pareto Front Overlay ===
subset = df.dropna(subset=["val_loss", "token_accuracy"])
points = subset[["val_loss", "token_accuracy"]].values
hull = ConvexHull(points)
pareto_points = points[hull.vertices]
pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=subset, x="token_accuracy", y="val_loss", hue="model_type", s=80)
plt.plot(pareto_points[:, 1], pareto_points[:, 0], "r--", label="Pareto Frontier")
plt.title("Pareto Frontier: Token Accuracy vs Validation Loss")
plt.xlabel("Token Accuracy")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/pareto_frontier_overlay.png")
plt.close()


# === Plot 2: KDE Jointplot ===
jp = sns.jointplot(
    data=subset, x="token_accuracy", y="val_loss",
    kind="kde", fill=True, cmap="mako", height=8
)
jp.plot_joint(sns.scatterplot, s=50, alpha=0.6)
jp.fig.suptitle("KDE Jointplot: Token Accuracy vs Validation Loss", y=1.02)
jp.fig.tight_layout()
jp.savefig(f"{output_dir}/jointplot_tokenacc_valloss.png")
plt.close()


# === Plot 3: Regression + Residuals ===
reg_df = df.dropna(subset=["val_loss", "embed_size", "activation"])
model = smf.ols("val_loss ~ embed_size + C(activation)", data=reg_df).fit()
reg_df["predicted_val_loss"] = model.predict(reg_df)
reg_df["residual"] = reg_df["val_loss"] - reg_df["predicted_val_loss"]

with open(os.path.join(output_dir, "regression_summary.txt"), "w") as f:
    f.write(str(model.summary()))

plt.figure(figsize=(8, 6))
sns.scatterplot(data=reg_df, x="predicted_val_loss", y="residual", hue="activation")
plt.axhline(0, color="gray", linestyle="--")
plt.title("Residuals vs Predicted Validation Loss")
plt.tight_layout()
plt.savefig(f"{output_dir}/regression_residuals.png")
plt.close()


# === Plot 4: Faceted val_loss vs embed_size ===
g = sns.lmplot(data=df, x='embed_size', y='val_loss', hue='model_type', col='activation',
               height=5, aspect=1.2, ci=None, scatter_kws={'s': 60})
g.set_titles(col_template="{col_name} activation")
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Validation Loss vs Embedding Size by Activation", fontsize=16)
g.savefig(f"{output_dir}/val_loss_vs_embed_size_by_activation.png")
plt.close()


# === Plot 5: Perplexity vs Accuracy (log scale) ===
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='token_accuracy', y='perplexity', hue='model_type', s=100)
plt.yscale('log')
plt.title("Perplexity vs Token Accuracy (log scale)")
plt.grid(True)
top = df.sort_values("token_accuracy", ascending=False).head(5)
for _, row in top.iterrows():
    plt.text(row['token_accuracy'], row['perplexity'], row['config_name'], fontsize=8)
plt.tight_layout()
plt.savefig(f"{output_dir}/perplexity_vs_accuracy_log.png")
plt.close()


# === Plot 6: Box + Strip by Activation ===
plt.figure(figsize=(10, 6))
order = df.groupby("activation")["val_loss"].median().sort_values().index
sns.boxplot(data=df, x='activation', y='val_loss', hue='model_type', order=order)
sns.stripplot(data=df, x='activation', y='val_loss', hue='model_type', dodge=True, alpha=0.5, order=order)
plt.title("Validation Loss by Activation Function (Ordered by Median)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/val_loss_by_activation_ordered.png")
plt.close()


# === Plot 7: Heatmaps of embed_size x k ===
if "k" in df.columns:
    for model in df["model_type"].dropna().unique():
        subset = df[df["model_type"] == model]
        if not subset.empty:
            heatmap_data = subset.pivot_table(index='embed_size', columns='k', values='val_loss', aggfunc='mean')
            plt.figure(figsize=(10, 6))
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
            plt.title(f"{model} - Validation Loss Heatmap (embed_size vs k)")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/val_loss_heatmap_{model}_embed_k.png")
            plt.close()


# === Plot 8: Correlation Matrix ===
plt.figure(figsize=(12, 10))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix (Hyperparams vs Metrics)")
plt.tight_layout()
plt.savefig(f"{output_dir}/correlation_matrix.png")
plt.close()


# === Leaderboard ===
top10 = df.sort_values("val_loss").head(10)
top10_path = f"{output_dir}/top10_val_loss.csv"
top10.to_csv(top10_path, index=False)
print(f"✅ Top 10 configs saved to: {top10_path}")
print("✅ All analysis complete.")