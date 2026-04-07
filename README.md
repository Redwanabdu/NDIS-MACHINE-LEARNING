# Network Intrusion Detection System (NIDS) using Machine Learning

## Overview
This project builds a machine learning-based Network Intrusion Detection System that classifies network traffic as normal or malicious (binary classification) and identifies specific attack types (multi-class classification).

## Dataset
- **NSL-KDD Dataset** (improved version of KDD Cup 1999)
- 125,973 training instances
- 41 features + 1 label
- Attack categories: DoS, Probe, R2L, U2R

## Results

### Binary Classification (Normal vs Attack)
| Metric | Score |
|--------|-------|
| Accuracy | 97.3% |
| Precision (abnormal) | 97% |
| Recall (abnormal) | 96% |
| F1-Score | 97% |

### Multi-Class Classification (Attack Types)
| Attack Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| DoS | 95% | 96% | 96% |
| Probe | 86% | 79% | 82% |
| R2L | 61% | 60% | 60% |
| Normal | 97% | 98% | 98% |

## Technologies Used
- Python 3.x
- pandas, numpy (data processing)
- scikit-learn (StandardScaler, SVM, train_test_split)
- Google Colab / Jupyter Notebook

## Key Implementation Steps
1. Data cleaning and preprocessing
2. Feature extraction using correlation analysis
3. Label mapping to attack categories (DoS, Probe, R2L, U2R)
4. Data normalization using StandardScaler
5. SVM with linear kernel for classification
6. Hyperparameter tuning
7. Performance evaluation (accuracy, precision, recall, F1)

## How to Run
```bash
# Install requirements
pip install -r requirements.txt

# Run the notebook
jupyter notebook nids_notebook.ipynb
