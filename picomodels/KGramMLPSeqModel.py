import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################
# 3. K-gram MLP in a sequence-to-sequence approach
################################################################################

class KGramMLPSeqModel(nn.Module):
    """
    For each position t in [0..seq_len-1], gather the last k tokens => embed => MLP => logits.
    Return (seq_len, batch, vocab_size).

    Potentially very large memory usage for big vocab or seq_len. chunk_size helps mitigate overhead.
    """

    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size

        self.embedding = nn.Embedding(vocab_size, embed_size)

        input_dim = k * embed_size
        layers = []

        # first layer
        layers.append(nn.Linear(input_dim, embed_size))
        layers.append(nn.LayerNorm(embed_size))    
        layers.append(nn.GELU())                       
        layers.append(nn.Dropout(p=0.1))   

        # Additional hidden layers
        for _ in range(num_inner_layers - 1):
            layers.append(nn.Linear(embed_size, embed_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p=0.1))

        # final layer
        layers.append(nn.Linear(embed_size, vocab_size))

        self.net = nn.Sequential(*layers)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        return: (seq_len, batch, vocab_size)
        We'll do a loop over time steps. chunk_size can reduce overhead.
        """
        seq_len, batch_size = tokens_seq.shape
        outputs = []

        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            block_outputs = []
            for t in range(start, end):
                batch_logits = []
                for b in range(batch_size):
                    if t < self.k:
                        needed = self.k - t
                        context_ids = [0]*needed + tokens_seq[:t, b].tolist()
                    else:
                        context_ids = tokens_seq[t-self.k:t, b].tolist()

                    context_tensor = torch.tensor(context_ids, device=tokens_seq.device).unsqueeze(0)  # (1, k)
                    context_embed = self.embedding(context_tensor)  # (1, k, embed_size)
                    context_flat = context_embed.view(1, -1) 
                    logits_b = self.net(context_flat)  # (1, vocab_size)
                    batch_logits.append(logits_b)
                block_outputs.append(torch.cat(batch_logits, dim=0).unsqueeze(0))  # (1, batch, vocab_size)

            block_outputs = torch.cat(block_outputs, dim=0)  # (chunk_size, batch, vocab_size)
            outputs.append(block_outputs)
            start = end

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, vocab_size)
        return outputs

