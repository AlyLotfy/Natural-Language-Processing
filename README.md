# 📝 English Text Summarization — Extractive vs. Abstractive

**A dual-approach NLP system combining graph-based TextRank and a fine-tuned BART transformer to summarize CNN/DailyMail news articles.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](#)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging_Face-FFD21E)](#)
[![Transformers](https://img.shields.io/badge/Transformers-BART-FF6F00)](#)
[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](#)

---

## 📌 Overview

This repository implements and benchmarks two **complementary** approaches to automatic text summarization:

| Approach | Model | What it does |
| --- | --- | --- |
| **Extractive** | TextRank (graph-based) | Picks the most central sentences from the source article |
| **Abstractive** | `facebook/bart-large-cnn` | Generates a new, paraphrased summary from scratch |

Both are evaluated on the **CNN/DailyMail** corpus against journalist-written reference highlights, using the standard NLP evaluation suite — **ROUGE-1, ROUGE-2, ROUGE-L** and **BLEU**.

The point of the project isn't just to beat a number — it's to make the trade-off **explicit**: extractive is interpretable and never hallucinates, abstractive is fluent and concise but can confabulate. Real-world summarization systems usually need both.

---

## 🎯 Why Two Approaches?

| | Extractive (TextRank) | Abstractive (BART) |
| --- | --- | --- |
| Output | Verbatim sentences from source | Newly composed text |
| Hallucination risk | None | Real (BART can invent facts) |
| Fluency / brevity | Limited by source phrasing | High |
| Compute cost | Tiny (graph + PageRank) | High (Transformer inference) |
| Best for | Compliance, legal, scientific | Marketing, news, casual reading |

Comparing them on the same corpus shows where each shines.

---

## 🧪 Approach

### Extractive — TextRank
- Sentence-level graph where edges weight pairwise similarity
- PageRank to rank sentences by centrality
- Top-K selection by score, reordered to preserve source flow

### Abstractive — BART
- Pre-trained `facebook/bart-large-cnn` (already fine-tuned on CNN/DailyMail)
- Standard generation parameters (beam search, length penalty)
- Optional fine-tuning hooks for domain adaptation

### Evaluation
- **ROUGE-1 / 2 / L** — n-gram recall vs. reference highlights
- **BLEU** — precision-oriented n-gram overlap
- Qualitative side-by-side comparison

---

## 📊 Sample Results

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
| --- | --- | --- | --- | --- |
| Extractive (TextRank) | 0.41 | 0.18 | 0.37 | 0.12 |
| Abstractive (BART) | **0.44** | **0.21** | **0.41** | **0.18** |

BART wins on every metric, which is unsurprising — but the gap is smaller than you'd expect, and TextRank is **orders of magnitude cheaper** to run.

---

## 🔧 Tech Stack

- **Python 3.7+**
- **NLTK** — sentence splitting & tokenization
- **NumPy** — TextRank similarity matrix
- **🤗 Transformers** — BART model & tokenizer
- **🤗 Datasets** — CNN/DailyMail loader
- **rouge-score** — ROUGE-1/2/L evaluation
- **tqdm** — progress bars

---

## 📁 Repository Layout

```
Natural-Language-Processing/
├── data/                    # Optional pre-downloaded dataset shards
├── src/
│   ├── extractive.py        # TextRank implementation
│   ├── abstractive.py       # BART inference
│   └── evaluate.py          # ROUGE & BLEU scoring
└── summarizer_show.py       # Demo: prints article + reference + both summaries
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/AlyLotfy/Natural-Language-Processing.git
cd Natural-Language-Processing

pip install nltk numpy transformers datasets rouge-score tqdm
python -m nltk.downloader punkt

# See both summaries side by side
python summarizer_show.py
```

---

## 📚 References

- Mihalcea & Tarau, 2004. *TextRank: Bringing Order into Texts.*
- Lewis et al., 2020. *BART: Denoising Sequence-to-Sequence Pre-training for NLG.*
- See et al., 2017. *Get To The Point: Summarization with Pointer-Generator Networks.*
- Hermann et al., 2015. *Teaching Machines to Read and Comprehend.* (CNN/DailyMail)
- Lin, 2004. *ROUGE: A Package for Automatic Evaluation of Summaries.*
- Papineni et al., 2002. *BLEU: a Method for Automatic Evaluation of Machine Translation.*
- Wolf et al., 2020. *Transformers: State-of-the-Art Natural Language Processing.*

---

## 📄 License

Academic project, AAST Computer Engineering.
