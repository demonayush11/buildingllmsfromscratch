# 🧠 LLMs From Scratch

A hands-on implementation of Large Language Models built from the ground up using Python and PyTorch — inspired by the book *"Build a Large Language Model (From Scratch)"* by Sebastian Raschka.

---

## 📌 What This Covers

This repo walks through the core building blocks of a GPT-style LLM, step by step:

### 1. 🔤 Tokenization
- Text preprocessing using regex-based splitting
- Building a vocabulary from raw text (`the-verdict.txt`)
- `SimpleTokenizerV1` — basic encode/decode
- `SimpleTokenizerV2` — with `<|unk|>` and `<|endoftext|>` special tokens
- Using **tiktoken** (GPT-2's tokenizer) for BPE tokenization

### 2. 📦 Data Pipeline
- `GPTDatasetV1` — sliding window dataset with configurable `max_length` and `stride`
- `create_dataloader_v1` — batched DataLoader for training

### 3. 🧮 Embeddings
- Token embeddings using `torch.nn.Embedding`
- Positional embeddings
- Combined input embeddings of shape `[batch, seq_len, embed_dim]`

### 4. 🎯 Self-Attention (from scratch)
- Manual dot-product attention scores
- Attention weight normalization (naive + softmax)
- Context vector computation
- Full attention matrix using `inputs @ inputs.T`
- Visualizing word embeddings in 3D

---

## 🛠️ Tech Stack

- Python 3.x
- PyTorch
- tiktoken
- Google Colab
- matplotlib (3D visualizations)

---

## 🚀 Getting Started

```bash
pip install torch tiktoken matplotlib
```

Then run the notebooks in order in Google Colab or locally.

---

## 📁 Structure
collab/
├── hello.py        # Entry point / scratch file
└── README.md       # You are here

---

## 📖 Reference

Based on [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka.