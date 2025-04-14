import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from collections import OrderedDict

# ✅ Keep merge_summaries() unchanged
def merge_summaries(analysis_dir="analysis_runs", output_csv="summary.csv"):
    merged_rows = OrderedDict()
    for fname in os.listdir(analysis_dir):
        if fname.startswith("summary_cache") and fname.endswith(".csv"):
            full_path = os.path.join(analysis_dir, fname)
            print(f"📥 Reading {fname}")
            with open(full_path, newline='', encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    config_name = row["config_name"]
                    merged_rows[config_name] = row
    if not merged_rows:
        print("⚠️ No summary_cache files found.")
        return
    output_path = os.path.join(analysis_dir, output_csv)
    keys = list(next(iter(merged_rows.values())).keys())
    with open(output_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(merged_rows.values())
    print(f"\n✅ Merged summary written to {output_path} ({len(merged_rows)} entries)")

# Merge summaries into one CSV
summary_path = "analysis_runs/summary.csv"
merge_summaries()

# Proceed only if summary file exists
if not os.path.exists(summary_path):
    raise FileNotFoundError("❌ summary.csv not found in analysis_runs/. Run analyze_all_checkpoints.py first.")

# Load and clean summary
df = pd.read_csv(summary_path)
numeric_cols = [
    'avg_loss', 'val_loss', 'perplexity', 'token_accuracy',
    'learning_rate', 'embed_size', 'k', 'chunk_size', 'inner_layers',
    'block_size', 'batch_size', 'tinystories_weight'
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Output plots folder
output_dir = "analysis_runs/plots"
os.makedirs(output_dir, exist_ok=True)

# 1. val_loss vs embed_size with trendlines and facets
g = sns.lmplot(data=df, x='embed_size', y='val_loss', hue='model_type', col='activation',
               height=5, aspect=1.2, ci=None, scatter_kws={'s': 60})
g.set_titles(col_template="{col_name} activation")
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Validation Loss vs Embedding Size by Activation", fontsize=16)
g.savefig(f"{output_dir}/val_loss_vs_embed_size_by_activation.png")
plt.close()

# 2. Perplexity vs Token Accuracy (log scale)
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

# 3. val_loss by activation (ordered + stripplot overlay)
plt.figure(figsize=(10, 6))
order = df.groupby("activation")["val_loss"].median().sort_values().index
sns.boxplot(data=df, x='activation', y='val_loss', hue='model_type', order=order)
sns.stripplot(data=df, x='activation', y='val_loss', hue='model_type', dodge=True, alpha=0.5, order=order)
plt.title("Validation Loss by Activation Function (Ordered by Median)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/val_loss_by_activation_ordered.png")
plt.close()

# 4. val_loss heatmaps by embed_size × k per model
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

# 5. Correlation Matrix
plt.figure(figsize=(12, 10))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix (Hyperparams vs Metrics)")
plt.tight_layout()
plt.savefig(f"{output_dir}/correlation_matrix.png")
plt.close()

# 6. Top 10 leaderboard
top10 = df.sort_values("val_loss").head(10)
top10_path = f"{output_dir}/top10_val_loss.csv"
top10.to_csv(top10_path, index=False)
print(f"✅ Top 10 configs saved to: {top10_path}")
