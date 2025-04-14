import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from collections import OrderedDict

def merge_summaries(analysis_dir="analysis_runs", output_csv="summary.csv"):
    merged_rows = OrderedDict()

    # Look for all cache files in analysis_runs/
    for fname in os.listdir(analysis_dir):
        if fname.startswith("summary_cache") and fname.endswith(".csv"):
            full_path = os.path.join(analysis_dir, fname)
            print(f"📥 Reading {fname}")

            with open(full_path, newline='', encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    config_name = row["config_name"]
                    merged_rows[config_name] = row  # Overwrite with the latest

    if not merged_rows:
        print("⚠️ No summary_cache files found.")
        return

    # Write merged file
    output_path = os.path.join(analysis_dir, output_csv)
    keys = list(next(iter(merged_rows.values())).keys())
    with open(output_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(merged_rows.values())

    print(f"\n✅ Merged summary written to {output_path} ({len(merged_rows)} entries)")


summary_path = "analysis_runs/summary.csv"
merge_summaries()
if not os.path.exists(summary_path):
    raise FileNotFoundError("❌ summary.csv not found in analysis_runs/. Run analyze_all_checkpoints.py first.")

# Load and clean
df = pd.read_csv(summary_path)
numeric_cols = [
    'avg_loss', 'val_loss', 'perplexity', 'token_accuracy',
    'learning_rate', 'embed_size', 'k', 'chunk_size', 'inner_layers',
    'block_size', 'batch_size', 'tinystories_weight'
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Create output folder
os.makedirs("analysis_runs/plots", exist_ok=True)

# 1. val_loss vs embed_size
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='embed_size', y='val_loss', hue='model_type', style='activation', s=100)
plt.title("Validation Loss vs Embedding Size")
plt.grid(True)
plt.tight_layout()
plt.savefig("analysis_runs/plots/val_loss_vs_embed_size.png")
plt.close()

# 2. perplexity vs token_accuracy
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='token_accuracy', y='perplexity', hue='model_type', s=100)
plt.title("Perplexity vs Token Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("analysis_runs/plots/perplexity_vs_accuracy.png")
plt.close()

# 3. val_loss by activation
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='activation', y='val_loss', hue='model_type')
plt.title("Validation Loss by Activation Function")
plt.grid(True)
plt.tight_layout()
plt.savefig("analysis_runs/plots/val_loss_by_activation.png")
plt.close()

# 4. val_loss heatmap by embed_size × k
if "k" in df.columns:
    heatmap_data = df.pivot_table(index='embed_size', columns='k', values='val_loss', aggfunc='mean')
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Validation Loss Heatmap (embed_size vs k)")
    plt.tight_layout()
    plt.savefig("analysis_runs/plots/val_loss_heatmap_embed_k.png")
    plt.close()

print("✅ Summary plots saved to: analysis_runs/plots/")
# 5. val_loss heatmap by embed_size × chunk_size