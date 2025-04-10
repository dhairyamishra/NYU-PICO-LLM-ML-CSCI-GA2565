import os
import re
import argparse
import torch
import matplotlib.pyplot as plt
from main import KGramMLPSeqModel, LSTMSeqModel, TransformerModel, generate_text
from torch.nn.functional import cosine_similarity
import tiktoken

def load_model(model_type, vocab_size, checkpoint_path, embed_size, k=3, chunk_size=1, num_inner_layers=1, block_size=128):
    if model_type == "kgram_mlp_seq":
        model = KGramMLPSeqModel(
            vocab_size, k=k, embed_size=embed_size,
            num_inner_layers=num_inner_layers, chunk_size=chunk_size
        )
    elif model_type == "lstm_seq":
        model = LSTMSeqModel(vocab_size, embed_size=embed_size, hidden_size=embed_size)
    elif model_type == "kvcache_transformer":
        model = TransformerModel(
            vocab_size=vocab_size, d_model=embed_size, n_heads=4,
            n_blocks=6, max_seq_len=block_size
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model

def analyze_checkpoints(checkpoint_dir_main, checkpoint_dir_sub, model_type, prompt, embed_size, k, chunk_size, inner_layers, block_size):
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    full_path = os.path.join(checkpoint_dir_main, checkpoint_dir_sub)
    checkpoint_files = sorted(
        [f for f in os.listdir(full_path) if f.startswith("epoch_") and f.endswith(".pt")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    generations = []
    for ckpt in checkpoint_files:
        path = os.path.join(full_path, ckpt)
        model = load_model(model_type, vocab_size, path, embed_size, k, chunk_size, inner_layers, block_size)
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
    loss_log_path = os.path.join(args.checkpoint_dir_main,args.checkpoint_dir_sub, "loss_log.pt")
    if os.path.exists(loss_log_path):
        loss_dict = torch.load(loss_log_path)
        epochs = sorted([int(k.split("_")[1]) for k in loss_dict.keys()])
        losses = [loss_dict[f"epoch_{e}"] for e in epochs]

        plt.figure()
        plt.plot(epochs, losses, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Avg Training Loss")
        plt.title(f"Loss Curve for {args.model_type}")
        plt.grid(True)
        plt.show()
    else:
        print("No loss_log.pt found. Skipping loss plot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir_main", default="checkpoints/", type=str, help="Path to model checkpoint folder")
    parser.add_argument("--checkpoint_dir_sub", 
                        
                        default="kgram_mlp_seq_tsw0.5_bs16_lr0.001_ep10_mlp1_k3_cs1_blk256_emb256_20250410_134230", 
                        
                        type=str, help="Path to specific models epock folder"
                        )
    parser.add_argument("--model_type", default="", type=str, choices=["kgram_mlp_seq", "lstm_seq", "kvcache_transformer"])
    parser.add_argument("--prompt", type=str, default="Once upon a", help="Prompt for generation")
    parser.add_argument("--embed_size", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--inner_layers", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    args = parser.parse_args()
    # --- Regex extraction from checkpoint_dir_sub ---
    filename = args.checkpoint_dir_sub
    patterns = {
        "model_type": r"^(kgram_mlp_seq|lstm_seq|kvcache_transformer)",
        "inner_layers": r"mlp(\d+)",
        "k": r"k(\d+)",
        "chunk_size": r"cs(\d+)",
        "block_size": r"blk(\d+)",
        "embed_size": r"emb(\d+)"
    }

    # Only fill in values not explicitly set by the user
    for key, pattern in patterns.items():
        if getattr(args, key) is None or getattr(args, key) == "":
            match = re.search(pattern, filename)
            if match:
                value = match.group(1)
                if value.isdigit():
                    setattr(args, key, int(value))
                else:
                    setattr(args, key, value)
    analyze_checkpoints(
        args.checkpoint_dir_main,
        args.checkpoint_dir_sub,
        args.model_type,
        args.prompt,
        args.embed_size,
        args.k,
        args.chunk_size,
        args.inner_layers,
        args.block_size
    )
    loss_log_path = os.path.join(args.checkpoint_dir_main,args.checkpoint_dir_sub, "loss_log.pt")
    plotlosses( loss_log_path, args)
# Load loss log

# IMPLEMENTATION NOTE:
# ADD MORE METRICS INFO 



# Example usage:
# RUN THIS USING THE FOLLOWING COMMAND:
# python analyze_checkpoints.py --checkpoint_dir checkpoints/kvcache_transformer_tsw0.5_bs16_lr0.001_ep10_mlp1_k3_cs1_blk128_emb128_20250410_024139 --model_type kgram_mlp_seq --prompt "Once upon a" --embed_size 128 --k 3 --chunk_size 1 --inner_layers 1 --block_size 128