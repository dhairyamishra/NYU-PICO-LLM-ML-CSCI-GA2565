# 📚 NYU-PICO-LLM-ML-CSCI-GA2565

**Starter code and project scaffolding for NYU’s CSCI-GA.2565 Machine Learning course.**  
This project helps you build, analyze, and experiment with simplified versions of large language models (LLMs), including MLPs, LSTMs, and Transformer decoders, while exploring generation strategies such as top-p (nucleus) sampling.

---

## 🧠 Course Objective

The goal is to **demystify language models** by implementing their core building blocks from scratch. You are encouraged to modify and extend the starter code, understand each component deeply, and simulate real-world ML workflows—on minimal hardware.

---

## 🏗️ Key Components

### 1. **KGramMLPSeqModel**

A simple MLP-based sequence model that predicts the next token based on the last *k* tokens (i.e., a fixed-size context window). This forms a baseline to understand positional encoding, embeddings, and fully-connected reasoning.

- **ML Design Choices:**
  - Token indices are embedded into vectors via `nn.Embedding`.
  - Vectors are concatenated and passed through an MLP.
  - Suitable for small contexts (k = 3 to 5).
  - No recurrence or attention—purely feedforward.

---

### 2. **TransformerModel** (Decoder-only GPT-style)

An implementation of a decoder-only transformer architecture, inspired by models like GPT-2 and LLaMA.

- **ML Design Choices:**
  - Uses **positional embeddings** added to token embeddings.
  - Stacked transformer blocks with:
    - **Masked multi-head self-attention**
    - **LayerNorm** (or optionally RMSNorm)
    - **Residual connections**
  - Final output is a projection back to vocabulary space.

- **Motivation:**
  - Understand how transformers process sequences in parallel.
  - Explore how attention allows dynamic contextual weighting.

---

### 3. **Top-p (Nucleus) Sampling**

You extend `generate_text` to implement top-p sampling, a decoding strategy where you sample from the smallest subset of tokens whose cumulative probability exceeds *p*.

- **Why top-p?**
  - Greedy decoding often produces repetitive or dull output.
  - Top-p ensures diversity while avoiding sampling from very unlikely tokens.
  - Common in real-world text generation systems.

---

## 🛠️ Files & Structure

| File/Folder            | Description |
|------------------------|-------------|
| `main.py`              | Main entry point for training and testing models |
| `analyze_checkpoints.py` | Analyzes model weights over epochs (e.g., embeddings, loss) |
| `picomodels/`          | Contains all model classes: `KGramMLPSeqModel`, `TransformerModel`, etc. |
| `checkpoints/`         | Saved model states for loading/resuming training |
| `oldcode/`             | Legacy scripts for reference or backup |
| `documents/`           | Possibly lecture notes or reference PDFs |
| `ReadMeImages/`        | Images used in README or reports |
| `.gitignore`           | Standard ignore file to avoid committing logs/checkpoints |

---

## 🧪 Core Tasks

1. **Run the LSTM Starter Code**
   - Start with `main.py` and `LSTMSeqModel` to validate your setup.
   - Use minimal configs like `--input_size 32` and `3seqs.txt` for quick iterations.

2. **Implement `KGramMLPSeqModel`**
   - Define an MLP architecture that takes a concatenated embedding of k tokens and predicts the next token.

3. **Implement Top-p Sampling**
   - Modify `generate_text` to apply cumulative probability truncation.

4. **Build `TransformerModel`**
   - Start with `torch.nn.Embedding`, stack self-attention blocks, and output logits over the vocabulary.
   - Use references like Karpathy’s [minGPT](https://github.com/karpathy/minGPT) as blueprints.

---

## 💻 Hardware Requirements

Designed for low-resource machines. Suggested settings:
- `--input_size 32`
- `--weight 0.0` (for regularization testing)
- Datasets like `3seqs.txt` for fast overfitting/testing

---

## 🎯 Learning Outcomes

By the end of this project, you will:
- Understand how modern LLMs are structured and trained
- Learn core architectural patterns: MLPs, LSTMs, Transformers
- Implement generation techniques like greedy vs top-p
- Analyze models using saved checkpoints and embeddings

---

## 📚 References

- [minGPT (Karpathy)](https://github.com/karpathy/minGPT)
- [LLaMA 3 GitHub](https://github.com/facebookresearch/llama)
- [NYU CSCI-GA.2565 Course Materials](https://cs.nyu.edu/~dsontag/courses/ml2020/)

---

## 🚀 Getting Started

```bash
# Setup environment (optional)
python -m venv env
source env/bin/activate

# Install PyTorch, etc.
pip install -r requirements.txt

# Run LSTM starter
python main.py --model_type lstm --input_size 32

# Train your custom MLP
python main.py --model_type kgram --k 4

# Train transformer
python main.py --model_type transformer --num_layers 4
