import math
import torch
import torch.nn as nn
import torch.nn.functional as F
################################################################################
# 5. Our "stub" Transformer with KV-cache 
#    Very slow Python loop for training. Multi-head sums head outputs.
################################################################################

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / norm)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Query, key, value projections
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Causal mask buffer
        self.register_buffer("mask", None)

    def forward(self, x, cache=None):
        """
        x: (seq_len, batch, d_model)
        cache: (optional) a dict for KV caching during inference.
          In inference mode, assume x has shape (1, batch, d_model).
        """
        seq_len, batch_size, _ = x.size()
        
        # If a cache is provided and we're in inference (single-token) mode, use KV-cache.
        if cache is not None and seq_len == 1:
            # Inference with KV-cache: compute new keys/values and update cache.
            q = self.query(x)   # (1, batch, d_model)
            k_new = self.key(x) # (1, batch, d_model)
            v_new = self.value(x) # (1, batch, d_model)
            # If cache exists, concatenate new values along the sequence dimension.
            if "k" in cache:
                k = torch.cat([cache["k"], k_new], dim=0)  # (cached_seq_len+1, batch, d_model)
                v = torch.cat([cache["v"], v_new], dim=0)
            else:
                k = k_new
                v = v_new
            # Update cache.
            cache["k"] = k
            cache["v"] = v
            # Reshape for multi-head attention.
            q = q.view(seq_len, batch_size, self.n_heads, self.head_dim)
            k = k.view(-1, batch_size, self.n_heads, self.head_dim)  # (-1, batch, n_heads, head_dim)
            v = v.view(-1, batch_size, self.n_heads, self.head_dim)
            # Permute: (batch, n_heads, seq_len, head_dim)
            q = q.permute(1, 2, 0, 3)   # (batch, n_heads, 1, head_dim)
            k = k.permute(1, 2, 0, 3)   # (batch, n_heads, cached_seq_len+1, head_dim)
            v = v.permute(1, 2, 0, 3)
            # Scaled dot-product attention.
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch, n_heads, 1, cached_seq_len+1)
            # In inference with caching, all previous tokens are valid, so we do not apply an additional causal mask here.
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_probs, v)  # (batch, n_heads, 1, head_dim)
            # Reshape back.
            attn_output = attn_output.permute(2, 0, 1, 3).reshape(seq_len, batch_size, self.d_model)
            output = self.out_proj(attn_output)
            return output

        # Otherwise, for training or when processing a full sequence, fall back to the original implementation.
        if self.mask is None or self.mask.size(0) != seq_len:
            # Create causal mask (upper triangular, excluding diagonal)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            self.register_buffer("mask", mask)
        
        q = self.query(x)  # (seq_len, batch, d_model)
        k = self.key(x)    # (seq_len, batch, d_model)
        v = self.value(x)  # (seq_len, batch, d_model)
        
        q = q.view(seq_len, batch_size, self.n_heads, self.head_dim)
        k = k.view(seq_len, batch_size, self.n_heads, self.head_dim)
        v = v.view(seq_len, batch_size, self.n_heads, self.head_dim)
        
        combined = torch.zeros(seq_len, batch_size, self.d_model, device=x.device)
        
        # Compute attention outputs per head using nested loops (slow but retained from your original style)
        for h in range(self.n_heads):
            head_dim = self.head_dim
            q_h = q[:, :, h, :]  # (seq_len, batch, head_dim)
            k_h = k[:, :, h, :]  # (seq_len, batch, head_dim)
            v_h = v[:, :, h, :]  # (seq_len, batch, head_dim)
            
            for b in range(batch_size):
                q_hb = q_h[:, b]  # (seq_len, head_dim)
                k_hb = k_h[:, b]  # (seq_len, head_dim)
                v_hb = v_h[:, b]  # (seq_len, head_dim)
                
                attn_scores = torch.matmul(q_hb, k_hb.transpose(0, 1)) / (head_dim ** 0.5)  # (seq_len, seq_len)
                attn_scores = attn_scores.masked_fill(self.mask, float('-inf'))
                attn_probs = F.softmax(attn_scores, dim=-1)  # (seq_len, seq_len)
                head_output = torch.matmul(attn_probs, v_hb)  # (seq_len, head_dim)
                
                combined[:, b, h * head_dim:(h + 1) * head_dim] = head_output
        
        output = self.out_proj(combined)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        # A simple two-layer MLP with GELU activation and a residual connection
        return x + self.linear2(F.gelu(self.linear1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ff = FeedForward(d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
    def forward(self, x, cache=None):
        # Pass the optional cache down to the attention layer.
        x = x + self.attn(self.norm1(x), cache=cache)
        x = x + self.ff(self.norm2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size=50257, d_model=512, n_heads=8, n_blocks=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # (a) Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional embeddings (using a fixed maximum sequence length)
        self.max_seq_len = 2048  # Adjust as needed
        self.pos_embedding = nn.Parameter(torch.zeros(self.max_seq_len, 1, d_model))
        # (b) Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_blocks)
        ])
        # Final layer normalization
        self.ln_f = RMSNorm(d_model)
        # (c) Output projection (unembedding)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, tokens_seq, cache=None):
        """
        tokens_seq: (seq_len, batch)
        cache: (optional) a dictionary for key–value caching, to be used during inference.
        """
        seq_len, batch_size = tokens_seq.size()
        
        # Get token embeddings.
        x = self.embedding(tokens_seq)  # (seq_len, batch, d_model)
        # Add positional embeddings (limited to the current sequence length).
        x = x + self.pos_embedding[:seq_len, :, :]
        
        # Pass through transformer blocks, optionally using cache.
        for block in self.blocks:
            x = block(x, cache=cache)
        
        x = self.ln_f(x)
        logits = self.output_projection(x)  # (seq_len, batch, vocab_size)
        
        return logits
