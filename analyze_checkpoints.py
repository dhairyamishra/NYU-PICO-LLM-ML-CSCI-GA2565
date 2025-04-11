import os
import re
import argparse
import torch
import matplotlib.pyplot as plt
from main import KGramMLPSeqModel, LSTMSeqModel, TransformerModel, generate_text, get_activation, get_model_config
from torch.nn.functional import cosine_similarity
import tiktoken

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
    
    print("\n=== Generation Samples Across Epochs ===")
    for name, gen, ano in generations:
        print(f"\n[{name}]")
        print("Generated text:", gen)
        # print("Annotated text:", ano)
    
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


        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.flatten()

        axs[0].plot(epochs, train_losses, marker='o', label="Train Loss")
        axs[0].plot(epochs, val_losses, marker='o', label="Validation Loss")
        axs[0].set_title("Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(epochs, accuracies, marker='o', color='green')
        axs[1].set_title("Token Accuracy")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        axs[1].grid(True)

        axs[2].plot(epochs, perplexities, marker='o', color='orange')
        axs[2].set_title("Perplexity")
        axs[2].set_xlabel("Epoch")
        axs[2].set_ylabel("Perplexity")
        axs[2].grid(True)

        axs[3].plot(epochs, lrs, marker='o', color='purple')
        axs[3].set_title("Learning Rate")
        axs[3].set_xlabel("Epoch")
        axs[3].set_ylabel("LR")
        axs[3].grid(True)

        plt.suptitle(f"Training Metrics for {args.model_type}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plot_path = os.path.join(args.checkpoint_dir_sub, "metrics_curve.png")
        plt.savefig(plot_path)
        print(f"Saved training metrics plot to {plot_path}")
        plt.show()
    else:
        print("No loss_log.pt found. Skipping loss plot.")




# lstm_seq_tsw0.5_bs16_lr0.001_ep10_mlp20_k3_cs1_blk256_emb256_20250410_142652
# kvcache_transformer_tsw0.5_bs16_lr0.001_ep10_mlp20_k3_cs1_blk256_emb256_20250410_144748
# kgram_mlp_seq_tsw0.5_bs16_lr0.001_ep10_mlp20_k3_cs1_blk256_emb256_20250410_140739
# kgram_mlp_seq_tsw0.5_bs16_lr0.005_ep10_mlp20_k3_cs1_blk256_emb256_20250410_154614 ---------WITH GELU
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir_sub", 
                        default=r"checkpoints\kgram_mlp_seq_tsw0.5_bs16_lr0.005_actgelu_ep5_mlp20_k3_cs1_blk128_emb128_20250410_194428", 
                        type=str, help="Path to specific models epock folder"
                        )
    parser.add_argument("--model_type", default="", type=str, choices=["kgram_mlp_seq", "lstm_seq", "kvcache_transformer"])
    parser.add_argument("--prompt", type=str, default="Once upon a", help="Prompt for generation")
    parser.add_argument("--embed_size", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--inner_layers", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu"], help="Activation function used in training")
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
        "activation": r"_act(relu|gelu)_"
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