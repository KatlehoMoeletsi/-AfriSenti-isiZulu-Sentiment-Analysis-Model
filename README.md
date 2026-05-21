# 🧠 AfriSenti isiZulu Sentiment Analysis Model

This project focuses on building a sentiment analysis model for African languages using the AfriSenti dataset. The main goal is to train a model that can predict whether a sentence written in isiZulu expresses a positive or negative emotion.

Sentiment analysis is important because it helps computers understand how people feel when they write messages, comments, or reviews. In this project, we focus specifically on African languages, which are often under-represented in machine learning research.

## 🎯 Objective
The objective of this project is to:

- 🔍 Analyse text written in isiZulu  
- 🧠 Train a machine learning model using the AfriSenti dataset  
- 😊 Predict whether a sentence expresses a positive or negative emotion  
- 📊 Evaluate the model using standard performance metrics  

---

This project uses the AfriSenti dataset, which contains labelled sentences from multiple African languages.

For this project, only the isiZulu part of the dataset is used. Each sentence in the dataset already has a sentiment label such as:

Positive
Negative
Project Structure

The repository is organised as follows:

## Project Structure
The repository is organised as follows:

```
## Project Structure
The repository is organised as follows:
SwahiliSentimentProject/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── baseline_models.ipynb
│   ├── transformers.ipynb
│
├── models/
│
├── results/
│   ├── graphs/
│   ├── reports/
│
├── src/
│   ├── augmentation.py
│   ├── preprocess.py
│   ├── train_baseline.py
│   ├── train_transformer.py
│   ├── evaluate.py
│   ├── explainability.py
│
├── requirements.txt
│
└── README.md
```


The model follows a simple natural language processing pipeline:

Load the dataset
Clean and preprocess the text
Convert the text into numerical features
Train a classification model
Predict sentiment (positive or negative)
Evaluate the model performance
Tools and Technologies

This project uses:

Python
Pandas
Scikit-learn
Jupyter Notebook
Natural Language Processing (NLP) techniques
Model Evaluation

The model is evaluated using the following metrics:

Accuracy
Precision
Recall
F1-score

These metrics help measure how well the model predicts positive and negative sentiment in isiZulu sentences.
