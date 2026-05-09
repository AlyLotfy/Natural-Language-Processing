# Airline Review Rating Prediction — NLP Pipeline

**Predicting customer satisfaction ratings from Singapore Airlines reviews using classical NLP: SVM with TF-IDF achieves 67.6 % accuracy on a 5-class problem.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](#)
[![NLTK](https://img.shields.io/badge/NLTK-154360)](#)
[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](#)

---

## Overview

This project builds an end-to-end NLP pipeline to predict a customer's 1–5 star rating from the text of their airline review. The dataset is 10,000 Singapore Airlines reviews scraped from a public review platform.

Multi-class sentiment classification (5 classes, heavily skewed toward 5-star reviews) is meaningfully harder than binary sentiment analysis. The pipeline handles the full workflow: noisy text in → structured predictions out.

---

## Dataset

- **Source:** Kaggle — Singapore Airlines customer reviews
- **Size:** 10,000 reviews
- **Target:** `rating` (integer, 1–5)
- **Class distribution:** Skewed — ~54 % of reviews are 5-star, making class imbalance a real challenge

---

## Pipeline

### 1. Text Cleaning

A multi-step cleaning function applied to every review:

- Lowercase conversion
- URL and number removal
- Punctuation stripping
- Tokenization (NLTK `word_tokenize`)
- Stopword removal
- Spell correction (`pyspellchecker`)
- Lemmatization (WordNet)

### 2. Feature Engineering

Two representations compared:

| Method | Implementation |
| --- | --- |
| TF-IDF | `TfidfVectorizer(ngram_range=(1,2))` |
| Bag of Words | `CountVectorizer(ngram_range=(1,2))` |

Bigrams were included in both to capture short phrases like "not good" or "very comfortable."

### 3. Models

Both models were tuned with `GridSearchCV` (5-fold cross-validation):

| Model | Grid searched |
| --- | --- |
| Support Vector Machine (SVC) | C, kernel {linear, rbf}, gamma |
| Random Forest | n_estimators, max_depth, min_samples_split |

### 4. Evaluation

Accuracy, precision, recall, F1-score per class, and confusion matrix.

---

## Results

| Model | Features | Accuracy |
| --- | --- | --- |
| **SVM** | **TF-IDF** | **67.6 %** |
| Random Forest | TF-IDF | 57.6 % |
| SVM | Bag of Words | 65.5 % |
| Random Forest | Bag of Words | 57.8 % |

SVM with TF-IDF is the best performer. The main difficulty is distinguishing 2, 3, and 4-star reviews — the model is confident at the extremes (1-star and 5-star) but uncertain in the middle.

Best SVM hyperparameters found: `C=1`, `kernel=linear`, `gamma=scale`.

---

## Key Takeaways

- TF-IDF consistently outperforms raw counts for this task.
- SVM beats Random Forest on text classification with sparse high-dimensional input — a known pattern.
- The class imbalance (5-star dominance) inflates overall accuracy; macro-averaged F1 is a better signal for this dataset.
- Spell checking during preprocessing measurably improved token quality but adds significant runtime.

---

## Tech Stack

- **Python 3.10+**
- **NLTK** — tokenization, stopwords, lemmatization
- **pyspellchecker** — spelling correction
- **scikit-learn** — TF-IDF, CountVectorizer, SVM, Random Forest, GridSearchCV
- **pandas / NumPy** — data handling
- **matplotlib / seaborn** — visualizations
- **Jupyter Notebook**

---

## Quick Start

```bash
git clone https://github.com/AlyLotfy/Natural-Language-Processing.git
cd Natural-Language-Processing

pip install nltk scikit-learn pyspellchecker pandas numpy matplotlib seaborn jupyter
python -m nltk.downloader stopwords punkt wordnet

# Place singapore_airlines_reviews.csv in the repo root
jupyter notebook
```

---

## Future Work

- Replace SVM with a fine-tuned BERT or DistilBERT model for contextual embeddings
- Weighted loss or oversampling to handle the class imbalance
- Aspect-based sentiment analysis to extract ratings per dimension (seat, food, service)

---

## License

Academic project, AAST College of Artificial Intelligence.
