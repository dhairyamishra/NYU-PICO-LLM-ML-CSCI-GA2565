# starter code by matus & o1-pro
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')  # ✅ fix for Unicode console printing
from datetime import datetime
import argparse
import time
import gc
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
# We do not import numpy or scikit-learn, so we implement a naive k-means in pure PyTorch.
# If you prefer scikit-learn, you can adapt the code.

from datasets import load_dataset
import tiktoken

################################################################################
# 1. Command-line arg parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple models on TinyStories and/or custom text files.")
    parser.add_argument("--input_files", nargs="*", default=None, help="Optional list of text files to mix in as data sources.")
    parser.add_argument("--tinystories_weight", type=float, default=0.5, help="Probability of sampling from TinyStories.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data to use for validation.")  
    parser.add_argument("--max_steps_per_epoch", type=int, default=50, help="Max steps per epoch.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-3, help="Learning rate.")
    parser.add_argument("--activation", type=str, choices=["relu", "gelu"], default="gelu",help="Activation function to use in models: 'relu' or 'gelu'.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--train_subset_size", type=int, default=10000, help="Number of TinyStories examples to load.")
    parser.add_argument("--log_interval_steps", type=int, default=10, help="Logging interval (in steps).")
    parser.add_argument("--sample_interval_seconds", type=int, default=60, help="Sampling interval (in seconds).")
    parser.add_argument("--num_inner_mlp_layers", type=int, default=20, help="Number of inner layers in k-gram MLP.")
    parser.add_argument("--kgram_k", type=int, default=3, help="Sliding window size for k-gram.")
    parser.add_argument("--kgram_chunk_size", type=int, default=2, help="Chunk size for k-gram processing.")
    parser.add_argument("--block_size", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--embed_size", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--prompt", type=str, default="Once upon a", help="Prompt for generation.")
    parser.add_argument("--device_id", type=str, default="cuda:0", help="Torch device ID (e.g., 'cuda:0').")
    parser.add_argument("--monosemantic_enabled", default=True, action="store_true", help="Enable monosemantic analysis.")
    parser.set_defaults(monosemantic_enabled=True)
    return parser.parse_args()

def get_activation(name):
    name = name.lower()
    if "relu" in name:
        return nn.ReLU()
    elif "gelu" in name:
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation: {name}")


def get_model_config(model_name, args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_config_str = (
        f"{model_name}_"
        f"tsw{args.tinystories_weight}_"
        f"bs{args.batch_size}_"
        f"lr{args.learning_rate}_"
        f"act{args.activation}_"
        f"ep{args.num_epochs}_"
        f"mlp{args.num_inner_mlp_layers}_"
        f"k{args.kgram_k}_"
        f"cs{args.kgram_chunk_size}_"
        f"blk{args.block_size}_"
        f"emb{args.embed_size}_"
        f"{timestamp}"
    )
    return model_config_str
################################################################################
# 2. Data handling: entire sequences up to block_size => (seq_len, batch)
################################################################################

class MixedSequenceDataset(torch.utils.data.Dataset):
    """
    If we also sample  from other sources, we need to mix them in.
    We assume that the other sources are already preprocessed and tokenized.
    We store two lists of entire token sequences:
      - tinystories_seqs
      - other_seqs
    Each sequence is length <= block_size.

    During __getitem__, we randomly pick from one list or the other with probability p_tiny.
    Return that entire sequence as a 1D LongTensor.
    """
    def __init__(self, tinystories_seqs, other_seqs, p_tiny: float):
        super().__init__()
        self.tinystories_seqs = tinystories_seqs
        self.other_seqs = other_seqs
        self.p_tiny = p_tiny
        self.max_seq_len = 5000  # or use a constructor argument if needed

        self.has_tinystories = (len(self.tinystories_seqs) > 0)
        self.has_other = (len(self.other_seqs) > 0)

        self.total_length = len(self.tinystories_seqs) + len(self.other_seqs)
        if self.total_length == 0:
            raise ValueError("No data found! Both TinyStories and other sets are empty.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        r = random.random()
        if self.has_tinystories and self.has_other:
            if r < self.p_tiny:
                i = random.randint(0, len(self.tinystories_seqs) - 1)
                seq = self.tinystories_seqs[i]
            else:
                i = random.randint(0, len(self.other_seqs) - 1)
                seq = self.other_seqs[i]
        elif self.has_tinystories:
            i = random.randint(0, len(self.tinystories_seqs) - 1)
            seq = self.tinystories_seqs[i]
        else:
            i = random.randint(0, len(self.other_seqs) - 1)
            seq = self.other_seqs[i]

        return torch.tensor(seq, dtype=torch.long)


def seq_collate_fn(batch):
    """
    batch: list of 1D LongTensors of various lengths [<= block_size].
    1) find max length
    2) pad with zeros
    3) shape => (max_len, batch_size)
    """
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)

    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq

    return padded


################################################################################
# 3. K-gram MLP in a sequence-to-sequence approach
################################################################################

def compute_next_token_loss(logits, tokens):
    """
    logits: (seq_len, batch, vocab_size)
    tokens: (seq_len, batch)
    Next-token prediction => we shift target by 1.
    """
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    preds = logits[:-1, :, :]  # (seq_len-1, batch, vocab_size)
    gold = tokens[1:, :]       # (seq_len-1, batch)

    preds = preds.reshape(-1, vocab_size)
    gold = gold.reshape(-1)

    # 🔍 Sanity checks before loss computation
    if (gold < 0).any() or (gold >= vocab_size).any():
        print("🔥 Invalid token ID detected in gold targets!")
        print("Max token id:", gold.max().item(), "Vocab size:", vocab_size)
        print("Min token id:", gold.min().item())
        print("First few gold tokens:", gold[:10].tolist())
        raise ValueError("Invalid token index in target tensor for cross_entropy")

    assert preds.size(0) == gold.size(0), f"Mismatch: preds={preds.size()}, gold={gold.size()}"

    return F.cross_entropy(preds, gold)


# Replacing one-hot(SLOW) with nnEmbedding(FAST) for k-gram MLP
# 🧠 New Strategy
# Use an nn.Embedding layer: self.embedding = nn.Embedding(vocab_size, embed_dim)
# Collect k-gram contexts for each position using unfold
# Embed, flatten, and pass through the MLP in one big matrix operation.

class KGramMLPSeqModel(nn.Module):
    def __init__(self, vocab_size, k=3, embed_size=128, num_inner_layers=1, chunk_size=1, activation=nn.GELU()):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.chunk_size = chunk_size
        self.max_seq_len = 5000  # or use a constructor argument if needed

        self.embedding = nn.Embedding(vocab_size, embed_size)
        input_dim = (k) * (embed_size)
        hidden_dim = embed_size
        output_dim = vocab_size

        # Use RMSNorm after each Linear
        layers = [
            nn.Linear(input_dim, hidden_dim),
            RMSNorm(hidden_dim),
            activation
        ]
        for _ in range(num_inner_layers):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                RMSNorm(hidden_dim),
                activation
            ]

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        Returns: (seq_len, batch, vocab_size)
        """
        seq_len, batch_size = tokens_seq.shape

        # Pad on the left with zeros for k-1 steps
        padded = F.pad(tokens_seq, (0, 0, self.k - 1, 0), value=0)  # (seq_len + k - 1, batch)

        # Unfold k-grams: (seq_len, batch, k)
        kgrams = torch.stack([padded[i:i + seq_len] for i in range(self.k)], dim=2)

        # Embed tokens: (seq_len, batch, k, embed_dim)
        embedded = self.embedding(kgrams)

        # Flatten k * embed_dim: (seq_len * batch, k * embed_dim)
        flattened = embedded.permute(1, 0, 2, 3).reshape(batch_size * seq_len, -1)

        # Run MLP: (seq_len * batch, vocab_size)
        logits = self.net(flattened)

        # Reshape back to (seq_len, batch, vocab_size)
        return logits.view(batch_size, seq_len, -1).permute(1, 0, 2)


################################################################################
# 4. LSTM-based seq2seq
################################################################################

class LSTMSeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size=1024, hidden_size=1024, activation=nn.Identity()):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.max_seq_len = 5000  # or use a constructor argument if needed

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        => (seq_len, batch, vocab_size)
        """
        emb = self.embedding(tokens_seq)   # (seq_len, batch, embed)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(emb)           # (seq_len, batch, hidden)
        logits = self.linear(out)         # (seq_len, batch, vocab_size)
        return self.activation(logits)

################################################################################
# 5. Our "stub" Transformer with KV-cache 
#    Very slow Python loop for training. Multi-head sums head outputs.
#    Implementing optimised attention head with causal mask
################################################################################

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / norm)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, past_k=None, past_v=None):
        seq_len, batch_size, _ = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.view(seq_len, batch_size, self.n_heads, 3 * self.head_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # (batch, n_heads, seq_len, head_dim)
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        # Append to past if any
        if past_k is not None and past_v is not None:
            k = torch.cat([past_k, k], dim=2)  # concat along seq_len
            v = torch.cat([past_v, v], dim=2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.tril(torch.ones(attn_scores.shape[-2:], device=x.device))
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        # merge heads
        attn_out = attn_out.permute(2, 0, 1, 3).contiguous()  # (seq_len, batch, n_heads, head_dim)
        attn_out = attn_out.view(seq_len, batch_size, self.d_model)
        return self.out_proj(attn_out), k, v

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, activation):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            activation,
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, past_kv=None):
        past_k, past_v = past_kv if past_kv else (None, None)
        attn_out, k, v = self.attn(self.norm1(x), past_k=past_k, past_v=past_v)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, (k, v)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size=50257, d_model=512, n_heads=1, n_blocks=1, max_seq_len=512, activation=nn.GELU()):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, activation) for _ in range(n_blocks)
        ])
        self.norm_final = RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, past_kv_cache=None):
        seq_len, batch_size = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds model max_seq_len={self.max_seq_len}.")
        device = tokens.device

        tok_emb = self.token_emb(tokens)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(1).expand(seq_len, batch_size)
        pos_emb = self.pos_emb(pos_ids)
        x = tok_emb + pos_emb

        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kv_cache[i] if past_kv_cache else None
            x, kv = block(x, past_kv=past_kv)
            new_kv_cache.append(kv)

        x = self.norm_final(x)
        logits = self.output_proj(x)
        return logits, new_kv_cache

################################################################################
# 6. K-Means Monosemantic (DISABLED by default)
################################################################################

def monosemantic_analysis_for_token(token_id, model, monosomatics, enc, device="cpu", top_n=5):
    # Get the embedding matrix if the model has one
    if not hasattr(model, 'token_emb'):
        print("Model has no embedding layer. Skipping analysis.")
        return []

    token_embedding = model.token_emb.weight[token_id]  # (d_model,)
    all_embeddings = model.token_emb.weight  # (vocab_size, d_model)

    # Normalize
    token_embedding = F.normalize(token_embedding, dim=0)
    all_embeddings = F.normalize(all_embeddings, dim=1)

    # Cosine similarity
    similarities = torch.matmul(all_embeddings, token_embedding)  # (vocab_size,)
    values, indices = torch.topk(similarities, top_n + 1)  # +1 to skip self
    neighbors = []

    for i in range(1, top_n + 1):
        tid = indices[i].item()
        sim = values[i].item()
        neighbors.append((sim, tid))

    return neighbors

################################################################################
# 7. Single code path for text generation
################################################################################

def nucleus_sampling(logits, p=0.95, temperature=8.0, past_tokens=None, repetition_penalty=1.0):
    """
    Top-p (nucleus) sampling with temperature and repetition penalty.
    """
    logits = logits.clone()  # avoid modifying in-place

    # Apply repetition penalty
    if past_tokens and repetition_penalty != 1.0:
        for token_id in set(past_tokens):
            if 0 <= token_id < logits.size(0):  # valid token range
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty

    # Handle deterministic case
    if p >= 1.0 and temperature == 0:
        return torch.argmax(logits).item()

    if temperature != 1.0:
        logits = logits / temperature

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)

    cutoff_index = torch.searchsorted(cumulative_probs, p, right=True).item()
    top_indices = sorted_indices[:cutoff_index + 1]
    top_probs = probs[:cutoff_index + 1]
    top_probs = top_probs / (top_probs.sum() + 1e-8)

    sampled_index = torch.multinomial(top_probs, 1).item()
    return top_indices[sampled_index].item()


def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  monosemantic_info=None,
                  do_monosemantic=False):
    model.eval()

    with torch.no_grad():
        context_tokens = enc.encode(init_text)
        annotation_list = []

        # Only transformer has kv_cache and max_seq_len
        is_transformer = hasattr(model, 'blocks') and hasattr(model, 'forward')
        has_kv_cache = 'kv_cache' in model.forward.__code__.co_varnames if is_transformer else False
        max_seq_len = getattr(model, "max_seq_len", 128)  # fallback to 128 or args.block_size
        past_kv_cache = None

        for i in range(max_new_tokens):
            if is_transformer:
                # Use only the last token if caching
                # ✅ Always truncate to model.max_seq_len
                if has_kv_cache:
                    input_tokens = context_tokens[-1:]
                else:
                    input_tokens = context_tokens[-max_seq_len:]
                # 🧱 Ensure model never sees more than max_seq_len tokens
                assert len(input_tokens) <= model.max_seq_len, (
                    f"input_tokens ({len(input_tokens)}) > max_seq_len ({model.max_seq_len})"
                )
                seq_tensor = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(1)
                logits = model(seq_tensor) if not has_kv_cache else model(seq_tensor, past_kv_cache)
            else:
                # Non-transformer model gets full context every time
                seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
                logits = model(seq_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]  # discard extra outputs if any

            next_logits = logits[-1, 0, :]

            if top_p is None:
                chosen_token = torch.argmax(next_logits).item()
            else:
                chosen_token = nucleus_sampling(
                    next_logits,
                    p=top_p,
                    temperature=0.8,                # or expose this as a param
                    past_tokens=context_tokens,    # repetition tracking
                    repetition_penalty=1.5         # tweakable
                )

            context_tokens.append(chosen_token)
            # ✅ Truncate context to avoid overflow in future iterations
            if len(context_tokens) > model.max_seq_len:
                context_tokens = context_tokens[-model.max_seq_len:]

            if do_monosemantic and monosemantic_info is not None:
                neighbors = monosemantic_analysis_for_token(
                    chosen_token, model, monosemantic_info, enc, device=device, top_n=5
                )
                annotation_list.append((chosen_token, neighbors))
            else:
                annotation_list.append((chosen_token, []))

    final_text = enc.decode(context_tokens)
    prefix_text = enc.decode(context_tokens[:-max_new_tokens])
    annotated_text = prefix_text + "".join(
        enc.decode([tid]) + (f"[NN={[enc.decode([x[1]]) for x in neighs]}]" if neighs else "")
        for tid, neighs in annotation_list
    )
    return final_text, annotated_text

################################################################################
# 8. Training
################################################################################

def train_one_model(model,
                    loader,
                    val_loader,
                    epochs,
                    model_name,
                    device,
                    lr=1e-3,
                    log_steps=100,
                    sample_interval=30,
                    max_steps_per_epoch=None,
                    enc=None,
                    monosemantic_info=None,
                    prompt="Once upon a"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.25,
        patience=1,
    )
    start_time = time.time()
    next_sample_time = start_time
    global_step = 0
    args = parse_args()
    model_config_str = get_model_config(model_name, args)
    checkpoint_dir = os.path.join("checkpoints", model_config_str)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        partial_loss = 0.0
        partial_count = 0
        step_in_epoch = 0
        grad_norm_preclip_total = 0.0
        grad_norm_postclip_total = 0.0
        max_param_grad_total = 0.0

        for batch_idx, batch_tokens in enumerate(loader, start=1):
            step_in_epoch += 1
            global_step += 1

            batch_tokens = batch_tokens.to(device)

            logits = model(batch_tokens)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = compute_next_token_loss(logits, batch_tokens)

            optimizer.zero_grad()
            loss.backward()

            # --- Compute pre-clipping gradient stats ---
            grad_norm_preclip = 0.0
            max_param_grad = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm_preclip += p.grad.data.norm(2).item()**2
                    max_param_grad = max(max_param_grad, p.grad.data.abs().max().item())
            grad_norm_preclip = grad_norm_preclip**0.5

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # --- Compute post-clipping gradient norm ---
            grad_norm_postclip = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm_postclip += p.grad.data.norm(2).item()**2
            grad_norm_postclip = grad_norm_postclip**0.5

            optimizer.step()

            total_loss += loss.item()
            partial_loss += loss.item()
            partial_count += 1
            grad_norm_preclip_total += grad_norm_preclip
            grad_norm_postclip_total += grad_norm_postclip
            max_param_grad_total += max_param_grad

            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                      f"Partial Avg Loss: {avg_part_loss:.4f}")
                partial_loss = 0.0
                partial_count = 0

            current_time = time.time()
            if current_time >= next_sample_time and enc is not None:
                with torch.no_grad():
                    print(f"\n[{model_name}] Generating sample text (greedy) at epoch={epoch}, step={batch_idx}...")
                    text_greedy, ann_greedy = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=None,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Greedy Sample: {text_greedy}")
                    print(f" Annotated: {ann_greedy}\n")

                    print(f"[{model_name}] Generating sample text (top-p=0.95) at epoch={epoch}, step={batch_idx}...")
                    text_topp, ann_topp = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=0.95,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=0.95) Sample: {text_topp}")
                    print(f" Annotated: {ann_topp}\n")

                    print(f"[{model_name}] Generating sample text (top-p=1.0) at epoch={epoch}, step={batch_idx}...")
                    text_topp1, ann_topp1 = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=1.0,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=1.0) Sample: {text_topp1}")
                    print(f" Annotated: {ann_topp1}\n")

                next_sample_time = current_time + sample_interval
                model.train()

            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
                break

        avg_loss = total_loss / step_in_epoch
        # Compute training metrics
        token_preds = torch.argmax(logits[:-1], dim=-1)
        token_targets = batch_tokens[1:]
        token_acc = (token_preds == token_targets).float().mean().item()

        grad_norm = sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
        weight_norm = sum(p.data.norm(2).item()**2 for p in model.parameters())**0.5
        avg_grad_norm_preclip = grad_norm_preclip_total / step_in_epoch
        avg_grad_norm_postclip = grad_norm_postclip_total / step_in_epoch
        avg_max_grad = max_param_grad_total / step_in_epoch


        #    Validation loss computation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = val_batch.to(device)
                val_logits = model(val_batch)
                if isinstance(val_logits, tuple):
                    val_logits = val_logits[0]
                val_loss += compute_next_token_loss(val_logits, val_batch).item()
                val_steps += 1
        avg_val_loss = val_loss / val_steps
        print(f"[{model_name}] Validation Loss after epoch {epoch}: {avg_val_loss:.4f}")
        # Save per-epoch loss to file
        loss_log_path = os.path.join(checkpoint_dir, "loss_log.pt")
        if os.path.exists(loss_log_path):
            loss_dict = torch.load(loss_log_path)
        else:
            loss_dict = {}
        # ✅ include validation loss in logs
        loss_dict[f"epoch_{epoch}"] = {
            "avg_loss": avg_loss,
            "val_loss": avg_val_loss,
            "perplexity": math.exp(avg_loss),
            "token_accuracy": token_acc,
            "grad_norm": grad_norm_postclip,  # retains legacy field
            "grad_norm_preclip": avg_grad_norm_preclip,
            "grad_norm_postclip": avg_grad_norm_postclip,
            "max_param_grad": avg_max_grad,
            "weight_norm": weight_norm,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        # Add to loss_dict

        torch.save(loss_dict, loss_log_path)

        # Save model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[{model_name}] Saved checkpoint to: {checkpoint_path}")
        print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Loss: {avg_loss:.4f}")

        
        # 🔁 Step scheduler
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[{model_name}] Current learning rate: {current_lr}")

################################################################################
# 9. Main
################################################################################

def main():
    args = parse_args()
    activation_fn = get_activation(args.activation)
    # Additional local variables from arguments
    k = args.kgram_k
    chunk_size = args.kgram_chunk_size
    embed_size = args.embed_size
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    block_size = args.block_size
    train_subset_size = args.train_subset_size
    log_interval_steps = args.log_interval_steps
    sample_interval_seconds = args.sample_interval_seconds
    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers
    monosemantic_enabled = args.monosemantic_enabled

    # NEW: pick device from args.device_id, fallback to cpu if needed
    requested_device_id = args.device_id
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    print(f"Using device: {device}, block_size={block_size}, kgram_k={k}, chunk_size={chunk_size}, embed_size={embed_size}")

    ############################################################################
    # Data
    ############################################################################
    tinystories_seqs = []
    other_seqs = []

    if args.tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={args.tinystories_weight}...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        dataset = dataset.select(range(train_subset_size))
    else:
        print("TinyStories weight=0 => skipping TinyStories.")
        dataset = None

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")

    if dataset is not None:
        for sample in dataset:
            text = sample['text']
            tokens = enc.encode(text)
            tokens = tokens[:block_size] # truncate to block_size
            if len(tokens) > 0:
                tinystories_seqs.append(tokens)
        print(f"TinyStories sequences: {len(tinystories_seqs)}")

    if args.input_files:
        for filepath in args.input_files:
            print(f"Reading custom text file: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                tokens = enc.encode(line)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    other_seqs.append(tokens)
        print(f"Custom input files: {len(other_seqs)} sequences loaded.")
    else:
        print("No custom input files provided.")

    p_tiny = args.tinystories_weight
    if len(tinystories_seqs) == 0 and p_tiny>0:
        print("Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it.")
    combined_dataset = MixedSequenceDataset(
        tinystories_seqs=tinystories_seqs,
        other_seqs=other_seqs,
        p_tiny=p_tiny
    )

    # ✅ NEW: Split into train/val datasets
    val_size = int(args.val_split * len(combined_dataset))
    train_size = len(combined_dataset) - val_size
    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=seq_collate_fn
    )

    ############################################################################
    # Models
    ############################################################################
    kgram_model = KGramMLPSeqModel(
        vocab_size=vocab_size,
        k=k,
        embed_size=embed_size,
        num_inner_layers=num_inner_layers,
        chunk_size=chunk_size,
        activation=activation_fn
    ).to(device)

    lstm_model = LSTMSeqModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=embed_size,
        activation=activation_fn
    ).to(device)

    # ⚠️ Important:
    # Match embed_size and block_size to your CLI args so the dimensions are consistent with your training setup.
    transformer = TransformerModel(
        vocab_size=vocab_size,
        d_model=embed_size,         # match your CLI arg
        n_heads=4,                  # configurable
        n_blocks=6,                 # configurable
        max_seq_len=block_size,     # match sequence length from args
        activation=activation_fn
    ).to(device)

    models = {
        "kgram_mlp_seq": kgram_model,
        "lstm_seq": lstm_model,
        "kvcache_transformer": transformer,
    }


    ############################################################################
    # Train each model
    ############################################################################
    for model_name, model in models.items():
        monosemantic_info = model.token_emb.weight.detach() if monosemantic_enabled and hasattr(model, "token_emb") else None
        print(f"Training model: {model} with monosemantic info: {monosemantic_info is not None}")
        print(f"\n=== Training model: {model_name} ===")
        # try catch to train model and handle errors
        try:
            train_one_model(
                model=model,
                loader=train_loader,
                val_loader=val_loader,  # ✅ NEW
                epochs=num_epochs,
                model_name=model_name,
                device=device,
                lr=learning_rate,
                log_steps=log_interval_steps,
                sample_interval=sample_interval_seconds,
                max_steps_per_epoch=max_steps_per_epoch,
                enc=enc,
                monosemantic_info=monosemantic_info,
                prompt=args.prompt
            )

            # Final generation from the user-provided prompt (args.prompt).
            with torch.no_grad():
                # 1) Greedy
                text_greedy, ann_greedy = generate_text(
                    model, enc, args.prompt, max_new_tokens=20, device=device,
                    top_p=None,
                )
                # 2) top-p=0.95
                text_topp, ann_topp = generate_text(
                    model, enc, args.prompt, max_new_tokens=20, device=device,
                    top_p=0.95,
                )
                # 3) top-p=1.0 => full distribution random sampling
                text_topp1, ann_topp1 = generate_text(
                    model, enc, args.prompt, max_new_tokens=20, device=device,
                    top_p=1.0,
                )
        except Exception as e:
            print(f"Error training model {model_name}: {e}")
            continue
        # Print final samples
        print(f"[{model_name}] Final sample (greedy) from prompt: '{args.prompt}'")
        print(text_greedy)
        print(f"Annotated:\n{ann_greedy}\n")

        print(f"[{model_name}] Final sample (top-p=0.95) from prompt: '{args.prompt}'")
        print(text_topp)
        print(f"Annotated:\n{ann_topp}\n")

        print(f"[{model_name}] Final sample (top-p=1.0) from prompt: '{args.prompt}'")
        print(text_topp1)
        print(f"Annotated:\n{ann_topp1}")
        print("--------------------------------------------------")
        # save the model --depreciated, saving in more efficient
        # model_config_str = get_model_config(model_name, args)
        # final_dir = os.path.join("picomodels", model_config_str)
        # os.makedirs(final_dir, exist_ok=True)
        # final_path = os.path.join(final_dir, f"{model_name}.pt")
        # torch.save(model.state_dict(), final_path)
        # print(f"Model saved to {final_path}")
        torch.cuda.empty_cache()
        gc.collect()


    print("\n*** All models trained successfully! ***")


if __name__ == "__main__":
    main()