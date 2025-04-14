import os
import json
import csv
import re
import argparse
import torch
import matplotlib.pyplot as plt
from main import KGramMLPSeqModel, LSTMSeqModel, TransformerModel, generate_text, get_activation, get_model_config
from torch.nn.functional import cosine_similarity
import tiktoken
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

def load_model(model_type, vocab_size, checkpoint_path, embed_size, k=3, chunk_size=1, num_inner_layers=1, block_size=128, activation="gelu"):

    if model_type == "kgram_mlp_seq":

        model = KGramMLPSeqModel(
            vocab_size, k=k, embed_size=embed_size,
            num_inner_layers=num_inner_layers, chunk_size=chunk_size,
            activation=get_activation(activation)
        )
    elif model_type == "lstm_seq":
        model = LSTMSeqModel(vocab_size, embed_size=embed_size, hidden_size=embed_size,
            activation=get_activation(activation)
        )
    elif model_type == "kvcache_transformer":
        model = TransformerModel(
            vocab_size=vocab_size, d_model=embed_size, n_heads=4,
            n_blocks=6, max_seq_len=block_size,
            activation=get_activation(activation)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model

def analyze_checkpoints(checkpoint_dir_sub, model_type, prompt, embed_size, k, chunk_size, inner_layers, block_size, activation):
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    full_path = os.path.join(checkpoint_dir_sub)

    checkpoint_files = sorted(
        [f for f in os.listdir(full_path) if f.startswith("epoch_") and f.endswith(".pt")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    generations = []
    for ckpt in checkpoint_files:
        path = os.path.join(full_path, ckpt)

        model = load_model(model_type, vocab_size, path, embed_size, k, chunk_size, inner_layers, block_size, activation)

        out, ano = generate_text(model, 
                               enc, 
                               prompt, 
                               max_new_tokens=30,
                               top_p=0.95,
                               monosemantic_info=None,
                               do_monosemantic=True,)
        generations.append((ckpt, out, ano))
    
    # print("\n=== Generation Samples Across Epochs ===")
    for name, gen, ano in generations:
        # print(f"\n[{name}]")
        # print("Generated text:", gen)
        # print("Annotated text:", ano)
        # Save as JSONL
        jsonl_path = os.path.join(checkpoint_dir_sub, "generations.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
            for ckpt, gen, ano in generations:
                json.dump({"checkpoint": ckpt, "generation": gen, "annotation": ano}, f_jsonl)
                f_jsonl.write("\n")
        # print(f"Saved generations to {jsonl_path}")

        # Save as CSV
        csv_path = os.path.join(checkpoint_dir_sub, "generations.csv")
        with open(csv_path, "w", newline='', encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=["checkpoint", "generation", "annotation"])
            writer.writeheader()
            for ckpt, gen, ano in generations:
                writer.writerow({"checkpoint": ckpt, "generation": gen, "annotation": ano})
        # print(f"Saved generations to {csv_path}")
    
    return generations

def plotlosses(loss_log_path, args):
    import matplotlib.pyplot as plt
    loss_log_path = os.path.join(args.checkpoint_dir_sub, "loss_log.pt")
    if os.path.exists(loss_log_path):
        loss_dict = torch.load(loss_log_path)
        epochs = sorted([int(k.split("_")[1]) for k in loss_dict.keys()])
        train_losses = [loss_dict[f"epoch_{e}"].get("avg_loss", float("nan")) for e in epochs]
        val_losses = [loss_dict[f"epoch_{e}"].get("val_loss", float("nan")) for e in epochs]
        accuracies = [loss_dict[f"epoch_{e}"].get("token_accuracy", float("nan")) for e in epochs]
        perplexities = [loss_dict[f"epoch_{e}"].get("perplexity", float("nan")) for e in epochs]
        lrs = [loss_dict[f"epoch_{e}"].get("learning_rate", float("nan")) for e in epochs]
        grad_norms = [loss_dict[f"epoch_{e}"].get("grad_norm", float("nan")) for e in epochs]
        weight_norms = [loss_dict[f"epoch_{e}"].get("weight_norm", float("nan")) for e in epochs]
        grad_norms_pre = [loss_dict[f"epoch_{e}"].get("grad_norm_preclip", float("nan")) for e in epochs]
        grad_norms_post = [loss_dict[f"epoch_{e}"].get("grad_norm_postclip", float("nan")) for e in epochs]
        max_param_grads = [loss_dict[f"epoch_{e}"].get("max_param_grad", float("nan")) for e in epochs]
        fig, axs = plt.subplots(2, 3, figsize=(16, 9))
        axs = axs.flatten()

        # 1. Loss
        axs[0].plot(epochs, train_losses, marker='o', label="Train Loss")
        axs[0].plot(epochs, val_losses, marker='o', label="Val Loss")
        axs[0].set_title("Loss (Train vs Val)")
        axs[0].legend()
        axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss"); axs[0].grid(True)

        # 2. Accuracy + Perplexity (normalized or separate y-axis)
        # Accuracy & Perplexity (merged using twin y-axis)
        ax1 = axs[1]
        ax2 = ax1.twinx()

        # Accuracy on left
        acc_line = ax1.plot(epochs, accuracies, marker='o', color='green', label='Accuracy')[0]
        ax1.set_ylabel("Accuracy", color='green')
        ax1.tick_params(axis='y', labelcolor='green')

        # Perplexity on right
        ppl_line = ax2.plot(epochs, perplexities, marker='o', color='orange', label='Perplexity')[0]
        ax2.set_ylabel("Perplexity", color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        # Shared X
        ax1.set_xlabel("Epoch")
        ax1.set_title("Token Accuracy & Perplexity")
        ax1.grid(True)

        # Combined legend
        lines = [acc_line, ppl_line]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        # 3. LR
        axs[2].plot(epochs, lrs, marker='o', color='purple')
        axs[2].set_title("Learning Rate")
        axs[2].set_xlabel("Epoch"); axs[2].set_ylabel("LR"); axs[2].grid(True)

        # 4. Grad Norms (Pre/Post Clip)
        axs[3].plot(epochs, grad_norms_pre, marker='o', label="Pre-clip", color="magenta")
        axs[3].plot(epochs, grad_norms_post, marker='o', label="Post-clip", color="red")
        axs[3].set_title("Gradient Norms (Pre vs Post Clip)")
        axs[3].legend()
        axs[3].set_xlabel("Epoch"); axs[3].set_ylabel("Grad Norm"); axs[3].grid(True)

        # 5. Max Gradient
        axs[4].plot(epochs, max_param_grads, marker='o', color="darkorange")
        axs[4].set_title("Max Param Gradient")
        axs[4].set_xlabel("Epoch"); axs[4].set_ylabel("Max |grad|"); axs[4].grid(True)

        # 6. Weight Norm
        axs[5].plot(epochs, weight_norms, marker='o', color="blue")
        axs[5].set_title("Weight Norm (L2)")
        axs[5].set_xlabel("Epoch"); axs[5].set_ylabel("L2 Norm"); axs[5].grid(True)


        plt.suptitle(
            f"{args.model_type} | act={args.activation} | emb={args.embed_size} | k={args.k} | cs={args.chunk_size} | mlp={args.inner_layers} | blk={args.block_size} | lr={args.learning_rate} | bs={args.batch_size} | tsw={args.tinystories_weight}",
            fontsize=14
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plot_path = os.path.join(args.checkpoint_dir_sub, "metrics_curve.png")
        plt.savefig(plot_path)
        # Target path for the plot
        print(f"Saved training metrics plot to {plot_path}")


    else:
        print("No loss_log.pt found. Skipping loss plot.")




# lstm_seq_tsw0.5_bs16_lr0.001_ep10_mlp20_k3_cs1_blk256_emb256_20250410_142652
# kvcache_transformer_tsw0.5_bs16_lr0.001_ep10_mlp20_k3_cs1_blk256_emb256_20250410_144748
# kgram_mlp_seq_tsw0.5_bs16_lr0.001_ep10_mlp20_k3_cs1_blk256_emb256_20250410_140739
# kgram_mlp_seq_tsw0.5_bs16_lr0.005_ep10_mlp20_k3_cs1_blk256_emb256_20250410_154614 ---------WITH GELU
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir_sub", 
                        default=r"checkpoints\kgram_mlp_seq_tsw0.8_bs32_lr0.001_actgelu_ep5_mlp10_k2_cs1_blk64_emb32_20250414_015747", 
                        type=str, help="Path to specific models epock folder"
                        )
    parser.add_argument("--model_type", default="", type=str, choices=["kgram_mlp_seq", "lstm_seq", "kvcache_transformer"])
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--tinystories_weight", type=float, default=None)
    parser.add_argument("--prompt", type=str, default="Once upon a", help="Prompt for generation")
    parser.add_argument("--embed_size", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--inner_layers", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu"], help="Activation function used in training")
    args = parser.parse_args()

    # --- Regex extraction from checkpoint_dir_sub ---
    filename = args.checkpoint_dir_sub.replace("\\", "/").split("/")[-1]
    print("Filename:", filename)
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

    for key, pattern in patterns.items():
        if getattr(args, key) is None or getattr(args, key) == "":
            match = re.search(pattern, filename)
            if match:
                value = match.group(1)
                if key in {"inner_layers", "k", "chunk_size", "block_size", "embed_size"}:
                    setattr(args, key, int(value))
                else:
                    setattr(args, key, value)

    # START ANALYSIS
    print("Analyzing checkpoints...")
    analyze_checkpoints(
        args.checkpoint_dir_sub,
        args.model_type,
        args.prompt,
        args.embed_size,
        args.k,
        args.chunk_size,
        args.inner_layers,
        args.block_size,
        args.activation
    )

    # Plot loss
    print("Plotting loss...")
    loss_log_path = os.path.join( args.checkpoint_dir_sub, "loss_log.pt")
    plotlosses(loss_log_path, args)

# IMPLEMENTATION NOTE:
# ADD MORE METRICS INFO 



# Example usage:
# RUN THIS USING THE FOLLOWING COMMAND:
# python analyze_checkpoints.py --checkpoint_dir checkpoints/kvcache_transformer_tsw0.5_bs16_lr0.001_ep10_mlp1_k3_cs1_blk128_emb128_20250410_024139 --model_type kgram_mlp_seq --prompt "Once upon a" --embed_size 128 --k 3 --chunk_size 1 --inner_layers 1 --block_size 128