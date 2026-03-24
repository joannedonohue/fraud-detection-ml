# Fraud Detection ML

Credit card fraud detection and Amazon review NLP using neural networks and ensemble machine learning.

---

## Overview

This repository contains three distinct machine learning projects applied to financial fraud detection and e-commerce sentiment analysis. The work explores a range of supervised and unsupervised techniques, with a focus on handling class imbalance and deploying multiple model architectures for comparison.

---

## Projects

### Dataset 1 — Advanced Fraud Detection (Simulated Credit Card Data)

**Goal:** Detect fraudulent transactions in a large-scale, highly imbalanced dataset.

**Dataset:** 1.3M training records (simulated credit card transactions)

**Techniques:**
- SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance
- Random Forest classifier
- TabNet (attention-based deep learning for tabular data)
- Neural network (TensorFlow/Keras)

**Key challenge:** Extreme class imbalance typical of real-world fraud scenarios; SMOTE used to synthetically balance minority (fraud) class before training.

---

### Dataset 2 — Credit Card Fraud Classification (UCI Dataset)

**Goal:** Compare algorithm performance on a benchmark fraud dataset with PCA-transformed features.

**Dataset:** UCI Credit Card Fraud dataset — 284,807 rows with anonymized PCA-engineered features

**Models evaluated:**
- Logistic Regression
- Random Forest
- Neural network

**Approach:** Evaluated models on both balanced and imbalanced versions of the data to assess sensitivity to class distribution.

---

### Dataset 3 — Amazon Reviews NLP Analysis

**Goal:** Extract sentiment signals and topics from unstructured product review text.

**Techniques:**
- Sentiment analysis: VADER and TextBlob
- Aspect-based sentiment extraction
- Topic modeling: LDA (Latent Dirichlet Allocation) with pyLDAvis visualization
- Logistic Regression classification on extracted features

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data manipulation | pandas, NumPy |
| Machine learning | scikit-learn |
| Deep learning | TensorFlow, Keras |
| Tabular DL | TabNet |
| NLP | NLTK, VADER, TextBlob |
| Topic modeling | LDA, pyLDAvis |
| Visualization | Matplotlib |

---

## Repository Structure

```
fraud-detection-ml/
├── fraud_detection_part1.ipynb       # Simulated dataset & TabNet/NN models
├── fraud_detection_part2.ipynb       # UCI dataset comparisons
├── Amazon-reviews/                   # NLP sentiment & topic modeling
├── Fraud_Detection_Presentation.pdf  # Project presentation slides
├── Supporting_Image_Problem2_Fraud.xlsx
└── README.md
```

---

## Outcomes

- Built a multi-model fraud detection pipeline handling 1.3M+ records with SMOTE balancing
- Demonstrated trade-offs between precision and recall across Random Forest, TabNet, and neural network architectures
- Applied VADER, TextBlob, and LDA to surface sentiment trends and latent topics from Amazon review text
- Produced comparative analysis of model performance on balanced vs. imbalanced data splits

---

## Getting Started

```bash
pip install tensorflow scikit-learn pandas nltk pyLDAvis pytorch-tabnet
jupyter notebook fraud_detection_part1.ipynb
```
