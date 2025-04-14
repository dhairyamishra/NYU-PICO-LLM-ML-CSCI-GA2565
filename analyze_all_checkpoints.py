import os
import re
import csv
import shutil
import torch
from datetime import datetime
from analyze_checkpoints import analyze_checkpoints, plotlosses

# --- Create timestamped output root ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ANALYSIS_DIR = os.path.join("analysis_runs", timestamp)
PLOTS_DIR = os.path.join(ANALYSIS_DIR, "plots")
GENERATIONS_DIR = os.path.join(ANALYSIS_DIR, "generations")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(GENERATIONS_DIR, exist_ok=True)

def extract_model_metadata(folder_name):
    patterns = {
        "model_type": r"^(kgram_mlp_seq|lstm_seq|kvcache_transformer)",
        "inner_layers": r"mlp(\d+)",
        "k": r"k(\d+)",
        "chunk_size": r"cs(\d+)",
        "block_size": r"blk(\d+)",
        "embed_size": r"emb(\d+)",
        "activation": r"_act(relu|gelu)_",
        "batch_size": r"bs(\d+)",
        "learning_rate": r"lr([0-9.]+)",
        "num_epochs": r"ep(\d+)",
        "tinystories_weight": r"tsw([0-9.]+)",
        "timestamp": r"_(\d{8}_\d{6})$"
    }
    metadata = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, folder_name)
        if match:
            val = match.group(1)
            if key in {"inner_layers", "k", "chunk_size", "block_size", "embed_size", "batch_size", "num_epochs"}:
                metadata[key] = int(val)
            elif key in {"learning_rate", "tinystories_weight"}:
                metadata[key] = float(val)
            else:
                metadata[key] = val
        else:
            metadata[key] = None
    return metadata
    
def summarize_loss_metrics(checkpoint_dir):
    loss_log_path = os.path.join(checkpoint_dir, "loss_log.pt")
    if not os.path.exists(loss_log_path):
        return None
    try:
        loss_dict = torch.load(loss_log_path)
        final_epoch = sorted(loss_dict.keys(), key=lambda k: int(k.split("_")[1]))[-1]
        entry = loss_dict[final_epoch]
        return {
            "val_loss": entry.get("val_loss", float("nan")),
            "perplexity": entry.get("perplexity", float("nan")),
            "token_accuracy": entry.get("token_accuracy", float("nan")),
            "learning_rate": entry.get("learning_rate", float("nan"))
        }
    except Exception as e:
        print(f"❌ Failed to read loss log in {checkpoint_dir}: {e}")
        return None

def copy_and_rename_plot(source_path, dest_name):
    if os.path.exists(source_path):
        dest_path = os.path.join(PLOTS_DIR, f"{dest_name}.png")
        shutil.copyfile(source_path, dest_path)
        return dest_path
    return None

def main():
    base_dir = "checkpoints"
    manifest_path = os.path.join(ANALYSIS_DIR, "checkpoint_manifest.csv")
    all_entries = []

    for folder in sorted(os.listdir(base_dir)):
        checkpoint_path = os.path.join(base_dir, folder)
        if not os.path.isdir(checkpoint_path):
            continue
        print(f"\n📊 Analyzing {folder}...")

        metadata = extract_model_metadata(folder)
        metrics = summarize_loss_metrics(checkpoint_path)

        if metrics:
            entry = {
                "checkpoint_folder": folder,
                **metadata,
                **metrics
            }
            all_entries.append(entry)

            # Plot and relocate plot file
            try:
                class Args:
                    checkpoint_dir_sub = checkpoint_path
                    model_type = metadata.get("model_type", "unknown")
                    activation = metadata.get("activation", "unknown")
                    embed_size = metadata.get("embed_size", 0)
                    k = metadata.get("k", 0)
                    chunk_size = metadata.get("chunk_size", 0)
                    inner_layers = metadata.get("inner_layers", 0)
                    block_size = metadata.get("block_size", 0)
                    batch_size = metadata.get("batch_size", 0)
                    learning_rate = metadata.get("learning_rate", 0.0)
                    tinystories_weight = metadata.get("tinystories_weight", 0.0)
                plotlosses(os.path.join(checkpoint_path, "loss_log.pt"), Args())
                copy_and_rename_plot(
                    os.path.join(checkpoint_path, "metrics_curve.png"),
                    folder
                )
            except Exception as e:
                print(f"⚠️ Skipped plotting for {folder}: {e}")


            # Optional: also extract generations
            try:
                prompt = "Once upon a"
                analyze_checkpoints(
                    checkpoint_path,
                    metadata["model_type"],
                    prompt,
                    metadata["embed_size"],
                    metadata["k"],
                    metadata["chunk_size"],
                    metadata["inner_layers"],
                    metadata["block_size"],
                    metadata["activation"]
                )

                # Move generations to central folder
                for ext in ["csv", "jsonl"]:
                    gen_src = os.path.join(checkpoint_path, f"generations.{ext}")
                    if os.path.exists(gen_src):
                        os.rename(gen_src, os.path.join(GENERATIONS_DIR, f"{folder}.{ext}"))

            except Exception as e:
                print(f"⚠️ Skipped generations for {folder}: {e}")

    # Write manifest CSV
    if all_entries:
        fieldnames = list(all_entries[0].keys())
        with open(manifest_path, "w", newline='', encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_entries)
        print(f"\n✅ Wrote manifest to: {manifest_path}")
    else:
        print("❌ No valid checkpoints found.")

if __name__ == "__main__":
    main()
    print("🔁 Running analysis script...")