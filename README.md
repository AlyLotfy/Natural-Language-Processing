# English Text Summarization

A dual-approach NLP system combining extractive TextRank and abstractive BART to generate concise summaries of CNN/DailyMail news articles, with automatic evaluation via ROUGE and BLE

## 📖 Project Description

This repository demonstrates two complementary approaches to automatic text summarization:

1. **Extractive Summarization**  
   - Graph-based TextRank algorithm.  
   - Selects the most central sentences from the source document.

2. **Abstractive Summarization**  
   - Pre-trained BART Transformer (`facebook/bart-large-cnn`).  
   - Generates concise, paraphrased summaries.

Both methods are applied to the CNN/DailyMail dataset. Summaries are compared against journalist‐written “highlights” using standard metrics (ROUGE-1/2/L, BLEU) and qualitative inspection.

---

## ⚙️ Requirements

- Python 3.7+  
- [NLTK](https://www.nltk.org/)  
- [NumPy](https://numpy.org/)  
- [Transformers](https://github.com/huggingface/transformers)  
- [Datasets](https://github.com/huggingface/datasets)  
- [rouge-score](https://github.com/google-research/google-research/tree/master/rouge)  
- [tqdm](https://github.com/tqdm/tqdm)  



## 📂 Repository Structure
.
-├── README.md

-├── data/

-│   └── (optional predownloaded dataset files)

-├── src/

-│   ├── extractive.py      # TextRank implementation

-│   ├── abstractive.py     # BART inference script

-│   └── evaluate.py        # ROUGE & BLEU scoring utilities

-└── summarizer_show.py     # Demo: prints article, reference, and both summaries

## 📈 Sample Results
-Method	ROUGE-1	ROUGE-2	ROUGE-L	BLEU

-Extractive	0.41	0.18	0.37	0.12

-Abstractive	0.44	0.21	0.41	0.18

## 📝 References
-Hermann et al., 2015. Teaching Machines to Read and Comprehend.

-See et al., 2017. Get To The Point: Summarization with Pointer-Generator Networks.

-Mihalcea & Tarau, 2004. TextRank: Bringing Order into Texts.

-Lewis et al., 2020. BART: Denoising Sequence-to-Sequence Pre-training for NLG.

-Lin, 2004. ROUGE: A Package for Automatic Evaluation of Summaries.

-Papineni et al., 2002. BLEU: a Method for Automatic Evaluation of MT.

-Wolf et al., 2020. Transformers: State-of-the-Art Natural Language Processing.
