# NYU-PICO-LLM-ML-CSCI-GA2565

Starter code and project scaffolding for NYU’s CSCI-GA.2565 Machine Learning course. Implement key components of modern language models, including MLPs, Transformer decoders, and nucleus sampling. Designed to mimic real-world ML workflows with flexibility to rewrite or extend the code. Runs on minimal hardware with simplified settings.

---

## 🧠 Course Context

This is not a template assignment—you’re encouraged to dive deep, understand the internals, and make your own engineering decisions. Whether you stick closely to the provided code or rewrite components, make sure you understand and can explain your approach.

## 💻 Hardware Requirements

You **don’t need high-end hardware**. The code can be run efficiently with smaller models and data. Use command-line flags like `--input_size 32`, `--weight 0.0`, and simplified files like `3seqs.txt` for faster and more memory-efficient runs.
![Sample Script:](./readmeImages/image.png)
---

## 🚀 Core Tasks

1. **Run the Starter Code**  
   Verify that the default LSTM model on TinyStories executes properly. If there are performance issues, apply aggressive memory-saving flags and simplified sequences. Take time to understand `torch.nn.Embedding` and why it's important in sequence modeling.

2. **Implement `KGramMLPSeqModel`**  
   Build a k-gram-based MLP model that performs sequence-to-sequence mapping. The `.forward()` method is scaffolded for you; ensure your network logic fits within it. Design your MLP architecture freely.

3. **Implement Nucleus (Top-p) Sampling**  
   Modify the `generate_text` function to include top-p sampling. This technique samples tokens based on a truncated probability distribution where cumulative probability reaches a threshold `p`. Compare different sampling behaviors.

4. **Implement `TransformerModel` (Decoder-only)**  
   Construct a GPT-style decoder-only transformer with:
   - `torch.nn.Embedding` input layer
   - Multiple transformer blocks with self-attention, skip connections, and normalization (e.g., LayerNorm or RMSNorm)
   - Final unembedding layer projecting to vocabulary size  
   Use blueprints from models like GPT-2 and LLaMA 3, or open-source references like [Karpathy’s `minGPT`](https://github.com/karpathy/minGPT).

---

## 📁 File Overview

- `main.py` – Entry point for model training and testing
- `models.py` – Contains model definitions (`KGramMLPSeqModel`, `TransformerModel`)
- `generate.py` – Text generation logic and sampling strategies
- `utils/` – Utility scripts for training and evaluation

---

## 🎯 Learning Goals

By completing this project, you will:

- Understand and implement core architectures used in modern LLMs
- Gain experience with embeddings, attention, and sampling techniques
- Develop practical intuition for real-world ML development pipelines

---

## 📚 References

- [minGPT by Karpathy](https://github.com/karpathy/minGPT)
- [LLaMA 3 GitHub](https://github.com/meta-llama/llama3)
- Lecture notes and materials from CSCI-GA.2565

---

Happy hacking! 🚀
