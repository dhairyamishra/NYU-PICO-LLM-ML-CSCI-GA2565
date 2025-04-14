# 🧠 NYU-PICO-LLM-ML-CSCI-GA2565

**Refactored & Enhanced Starter Code for NYU's CSCI-GA.2565 Machine Learning Course.**

This repo provides an extensible, scalable pipeline for training, analyzing, and visualizing K-Gram MLPs, LSTMs, and Transformer models on language generation tasks—complete with full logging, evaluation plots, and comparative metrics.

---

## 🆕 What's New Compared to Original Scaffolding?

| Feature | Original Scaffold | Enhanced Version |
|--------|-------------------|------------------|
| Model Types | KGramMLP, partial LSTM | ✅ LSTM, ✅ Transformer with KV cache, ✅ Fully implemented KGramMLP |
| Training | Single-model | ✅ Multi-model batch training with config tracking |
| Logging | Console-only | ✅ Loss, val_loss, perplexity, accuracy, gradients, LR over epochs |
| Analysis | Minimal | ✅ Comprehensive plotting & generation: `analyze_checkpoints.py`, `analyze_all_checkpoints.py`, `total_summary_analysis.py` |
| Generation | Greedy | ✅ Greedy + top-p + repetition penalty + annotations |
| Reproducibility | Manual | ✅ Deterministic training via hashed config names |
| Automation | None | ✅ Full auto pipeline with `auto_train_then_analyze.py` |

---

## 📚 Machine Learning Topics Covered (with Explanations)

### 🔢 Embeddings
**Concept:** Learn dense vector representations for discrete tokens. Used by all models.
- `nn.Embedding(vocab_size, embed_size)` maps token IDs to continuous vectors.

### 🔁 RNNs (LSTM)
**Model:** `LSTMSeqModel`
- Sequence modeling via recurrent loops
- Captures long-term dependencies using hidden and cell states
- Trained with teacher-forcing for next-token prediction

### 🧠 MLPs (K-Gram MLP)
**Model:** `KGramMLPSeqModel`
- Input: last `k` tokens → embedding → flatten → MLP → logits
- No recurrence or attention
- Lightweight baseline, good for understanding spatial vs sequential learning

### ⚡ Transformers
**Model:** `TransformerModel`
- Decoder-only, GPT-style
- Multi-head self-attention + causal masking + positional embeddings
- Includes KV cache for fast generation

### 🎲 Sampling Algorithms
**Implemented:** Greedy, Top-p (nucleus), Repetition Penalty
- `generate_text(...)` accepts `top_p`, `temperature`, and `past_tokens`
- Repetition penalty discourages repeating recent tokens

### 📉 Training Loss
**Function:** `compute_next_token_loss`
- Cross-entropy between predicted logits and ground truth
- Loss logged per step, epoch, and split (train/val)

### 📈 Training Metrics
Tracked in `loss_log.pt`:
- `avg_loss`, `val_loss`, `token_accuracy`, `perplexity`
- `grad_norm` (pre & post clip), `weight_norm`, `max_param_grad`

### 📊 Analysis & Evaluation
- Per-epoch generation (`generations.csv`, `generations.jsonl`)
- Training curve plots
- Global analysis across all runs (Pareto plots, regression models, heatmaps)

---

## 🔬 How to Use the Scripts (with Hypothetical Flow)

### 1. Train a Batch of Models
```bash
python batch_train.py
```
This will:
- Randomly sample hyperparameter combos (e.g., `embed_size=128`, `k=3`, `act=gelu`)
- Train each model for a few epochs
- Save checkpoints and logs to `checkpoints/<config_name>/`

### 2. Analyze All Checkpoints
```bash
python analyze_all_checkpoints.py --workers 8 --skip_existing
```
This will:
- Read each `checkpoints/` subdir
- Generate `generations.csv/jsonl`, `metrics_curve.png`
- Log final metrics to `analysis_runs/summary_cache.csv`

### 3. Run Full Pipeline
```bash
python auto_train_then_analyze.py --fast
```
This does both the above steps + logs output to timestamped files under `logs/`

### 4. Generate Research Plots
```bash
python total_summary_analysis.py
```
This generates plots to `analysis_runs/plots/`, including:
- Pareto frontier (val_loss vs token_accuracy)
- KDE jointplots
- Regression + residuals (e.g., `val_loss ~ embed_size + activation`)
- Boxplots by activation
- Heatmaps of val_loss over `embed_size × k`
- Correlation matrix of all numeric metrics

---

## 📂 Directory Structure

| Path                      | Purpose |
|---------------------------|---------|
| `main.py`                | Trains all models sequentially |
| `batch_train.py`         | Trains many model configs automatically |
| `analyze_checkpoints.py` | Analyze one run with plots & generations |
| `analyze_all_checkpoints.py` | Analyze all checkpoint dirs in bulk |
| `total_summary_analysis.py` | Create plots, correlations, regressions |
| `checkpoints/`           | Saved model checkpoints per epoch |
| `analysis_runs/`         | Generation samples + plots + metrics |
| `picomodels/`            | Final weights (optional) |
| `logs/`                  | Timestamped logs from training/analysis |

---

## 🧠 Example: How to Interpret Results

Let’s say we ran 100 configs. Some insights you might discover:

- **Pareto plot:** Transformer configs dominate high accuracy + low val_loss regions.
- **Jointplot:** Most configs cluster around val_loss ~2.0 and accuracy ~0.4.
- **Regression:** Increasing `embed_size` lowers `val_loss`, but only for GELU activations.
- **Boxplot:** LSTMs perform worse with ReLU than GELU.
- **Heatmap:** `k=1` is generally worse than `k=3` for MLPs.

---

## ✅ Conclusion
This project isn't just a starter codebase—it's a mini research lab. With structured training, deep metric logging, and automated comparative analysis, you can explore how LLMs behave under different configurations.

For questions or feature ideas, feel free to open an issue or pull request!

---

Happy training! 🧪
