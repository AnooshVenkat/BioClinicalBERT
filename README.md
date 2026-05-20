# Adverse SDoH Extraction using BioClinicalBERT

This project implements a multi-label classification pipeline to detect **Adverse Social Determinants of Health (SDoH)** from unstructured clinical notes (MIMIC-III). It utilizes **BioClinicalBERT** with specific strategies to handle the extreme data scarcity and class imbalance inherent in SDoH documentation.

## Project Overview

* **Goal:** Classify sentences into 6 adverse categories: `Employment`, `Housing`, `Transportation`, `Parental`, `Relationship`, and `Social Support`.
* **Model:** `emilyalsentzer/Bio_ClinicalBERT` (Pre-trained on MIMIC-III).
* **Key Challenge:** Extreme rarity of positive labels (e.g., Housing issues appear in <0.2% of real data).

## Key Methodologies

1.  **Transfer Learning & Regularization:**
    * freezes embeddings and the bottom 6 layers of BERT to preserve pre-trained medical knowledge.
    * Fine-tunes the top 6 layers and classifier head.
2.  **Handling Imbalance:**
    * **Synthetic Augmentation:** Merges AI-generated synthetic sentences with real training data.
    * **Cost-Sensitive Learning:** Uses `AsymmetricLoss`.

## Installation & Requirements

```bash
pip install torch transformers datasets pandas scikit-learn wandb numpy
```

## Data Structure

The pipeline requires two types of input data:

* **Real Data:** `SDOH_MIMICIII_physio_release.csv` (MIMIC-III subset).
* **Synthetic Data:** CSVs containing generated sentences (e.g., `SyntheticSentences_Round1.csv`).

> **Note:** Real data is split 60/20/20 (Train/Val/Test). Synthetic data is added **ONLY** to the Training set.

## Performance Results

| Metric | Score |
| :--- | :--- | 
| **Micro ROC-AUC** | **~0.93** | 
| **Micro F1** | **~0.44** | 

## Usage

Run the training script (ensure GPU is available):

```python
# The script performs the following steps:
# 1. Loads and cleans MIMIC-III data.
# 2. Augments training set with synthetic data.
# 3. Calculates class weights.
# 4. Fine-tunes BioClinicalBERT (Frozen layers).
```
