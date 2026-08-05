# 🧠 AfriSenti Swahili Sentiment Analysis using Machine Learning and Transformer Models

This project investigates sentiment analysis for **Swahili**, one of Africa's most widely spoken languages, using both traditional machine learning algorithms and state-of-the-art transformer models.

The objective is to determine whether a Swahili tweet expresses **positive**, **neutral**, or **negative** sentiment while evaluating whether African language-specific transformer models outperform conventional machine learning techniques.

---

## 📌 Project Overview

Natural Language Processing (NLP) has achieved remarkable success for English and other high-resource languages. However, African languages such as Swahili remain underrepresented due to limited annotated datasets and language resources.

This project addresses this challenge by benchmarking traditional machine learning models against transformer-based architectures using the **AfriSenti Swahili dataset**.

---

## 🎯 Objectives

The project aims to:

- Perform sentiment analysis on Swahili tweets
- Build an end-to-end NLP pipeline
- Compare traditional machine learning models with transformer models
- Evaluate whether African language pretraining improves performance
- Analyse model strengths and weaknesses through error analysis
- Establish a strong baseline for future Swahili NLP research

---

## 📂 Dataset

This project uses the **AfriSenti Swahili Dataset**, containing manually labelled Swahili tweets.

Sentiment labels include:

- 😀 Positive
- 😐 Neutral
- 😞 Negative

---

## 🚀 Project Pipeline

The workflow consists of:

1. Data Collection
2. Data Cleaning
3. Text Preprocessing
4. Feature Engineering
5. TF-IDF Vectorisation
6. Baseline Machine Learning Models
7. Transformer Fine-tuning
8. Model Evaluation
9. Error Analysis
10. Performance Comparison

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

- Remove URLs
- Remove Twitter mentions
- Remove punctuation
- Remove special characters
- Remove extra whitespace
- Text normalisation
- Preserve stopwords for improved sentiment classification

---

## 🤖 Models Implemented

### Baseline Machine Learning Models

- Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)

### Transformer Models

- XLM-RoBERTa
- AfriBERTa

---

## 📊 Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- Weighted F1-score
- Classification Report
- Confusion Matrix

---

## 📈 Key Findings

- Logistic Regression was the strongest traditional machine learning baseline.
- AfriBERTa achieved the best overall performance.
- African language-specific pretraining significantly outperformed generic multilingual transformers.
- XLM-RoBERTa underperformed compared to simpler machine learning models.
- Dataset imbalance negatively affected minority class prediction.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Hugging Face Transformers
- PyTorch
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 📁 Project Structure

```
SwahiliSentimentProject/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── baseline_models.ipynb
│   └── transformers.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── augmentation.py
│   ├── train_baseline.py
│   ├── train_transformer.py
│   ├── evaluate.py
│   └── explainability.py
│
├── models/
│
├── results/
│   ├── reports/
│   ├── graphs/
│   ├── confusion_matrices/
│   └── model_comparison/
│
├── requirements.txt
└── README.md
```

---

## 💡 Future Improvements

- Increase dataset size
- Apply data augmentation (Back Translation)
- Hyperparameter optimisation
- Cross-validation
- GPU training
- Deploy the model using FastAPI
- Build a Streamlit web application
- Dockerise the application
- CI/CD with GitHub Actions

---

## 📚 Skills Demonstrated

- Natural Language Processing
- Machine Learning
- Transformer Models
- Text Classification
- Feature Engineering
- Model Evaluation
- Error Analysis
- Python Development
- Data Preprocessing
- Research and Experimentation

---

## 📜 License

This project is released under the MIT License.
