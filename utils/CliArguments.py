import argparse
################################################################################
# 1. Command-line arg parsing
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files.")
    parser.add_argument("--input_files", nargs="*", default=None,
                        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).")
    parser.add_argument("--tinystories_weight", type=float, default=0.5,
                        help="Probability of sampling from TinyStories if present. Default=0.5. (set to 0.0 to skip TinyStories).")
    parser.add_argument("--max_steps_per_epoch", type=int, default=300,
                        help="If set, each epoch ends after this many steps (for quick tests).")
    parser.add_argument("--num_inner_mlp_layers", type=int, default=1,
                        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. Default=1.")
    parser.add_argument("--monosemantic_enabled", action="store_true",
                        help="(DISABLED BY DEFAULT) If set, run the monosemantic analysis.")
    parser.set_defaults(monosemantic_enabled=False)  # disable by default

    # Additional hyperparams to mitigate slow k-gram
    parser.add_argument("--kgram_k", type=int, default=3,
                        help="Sliding window size for k-gram MLP. Smaller can reduce memory usage. Default=3.")
    parser.add_argument("--kgram_chunk_size", type=int, default=1,
                        help="Process k-gram timesteps in micro-batches. Default=1.")
    
    parser.add_argument("--embed_size", type=int, default=512,
                        help="Dimension of the embedding layer for LSTM, MLP, etc. Default=1024.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size per epoch. Default=16.")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Epochs per training run. Default=3.")
    # New arguments:
    parser.add_argument("--block_size", type=int, default=512,
                        help="Maximum sequence length for each example. Default=1024.")
    parser.add_argument("--train_subset_size", type=int, default=512,
                        help="train_subset_size for each epoch. Default=512.")
    parser.add_argument("--log_interval_steps", type=int, default=100,
                        help="log_interval_steps ")
    parser.add_argument("--sample_interval_seconds", type=int, default=2,
                        help="print sample every sample_interval_seconds seconds. Default=2.")
    
    # New arguments:
    parser.add_argument("--prompt", type=str, default="Once upon a",
                        help="Prompt used for generation. Default='Once upon a'.")

    # Newly added device argument:
    parser.add_argument("--device_id", type=str, default="cuda:0",
                        help="Torch device identifier (default='cuda:0'). If CUDA is unavailable, fallback to 'cpu'.")
    # New arguments:
    args = parser.parse_args()
    return args

