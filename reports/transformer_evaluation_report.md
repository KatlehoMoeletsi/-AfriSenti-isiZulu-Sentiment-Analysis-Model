# Sentiment Analysis for Swahili - Transformer Models & Evaluation
## Student Contribution: Transformer Implementation, Model Comparison & Explainability

### 1. Methodology Overview
- **Baseline Models**: Naive Bayes, Logistic Regression
- **Transformer Model**: XLM-RoBERTa (multilingual)
- **Evaluation Metrics**: Accuracy, F1-Score, Precision, Recall

### 2. Dataset Description
| Metric | Value |
|--------|-------|
| Total Samples | 453 tweets |
| Classes | Negative (48), Neutral (268), Positive (137) |
| Language | Swahili |

### 3. Results
| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Naive Bayes | 0.5934 | 0.4420 | 0.3521 | 0.5934 |
| Logistic Regression | **0.6044** | **0.4662** | **0.6527** | 0.6044 |
| XLM-RoBERTa | 0.5600 | 0.4021 | 0.3136 | 0.5600 |

### 4. Key Findings
- **Best Model**: Logistic Regression (60.44% accuracy, 46.62% F1)
- Transformers underperformed due to limited data (453 samples)
- Class imbalance (only 48 negative samples) remains a challenge

### 5. Files Generated
| File | Location |
|------|----------|
| Complete pipeline | `full_analysis.py` |
| Best model | `models/logistic_regression.pkl` |
| Results table | `results/model_comparison.csv` |
| Visualization | `results/model_comparison_chart.png` |

---
*Report generated: May 28, 2026*
