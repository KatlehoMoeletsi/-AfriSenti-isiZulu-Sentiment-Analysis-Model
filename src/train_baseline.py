# ==========================================
# IMPORTS
# ==========================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# ==========================================
# CREATE FOLDERS IF THEY DON'T EXIST
# ==========================================

os.makedirs("models/baseline", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/processed/cleaned_data.csv"
)

print("\nDataset Loaded Successfully\n")

print(df.head())

# ==========================================
# CHECK CLASS DISTRIBUTION
# ==========================================

print("\nClass Distribution:\n")

print(df["label"].value_counts())

# ==========================================
# FEATURES + LABELS
# ==========================================

X = df["clean_text"]

y = df["label_encoded"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)

print("\nTrain-Test Split Completed\n")

# ==========================================
# TF-IDF VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer(

    max_features=5000,

    ngram_range=(1,2)
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

print("TF-IDF Vectorization Completed\n")

# ==========================================
# LOGISTIC REGRESSION
# ==========================================

print("Training Logistic Regression...\n")

lr_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

lr_model.fit(
    X_train_tfidf,
    y_train
)

lr_pred = lr_model.predict(
    X_test_tfidf
)

# ==========================================
# NAIVE BAYES
# ==========================================

print("Training Naive Bayes...\n")

nb_model = MultinomialNB()

nb_model.fit(
    X_train_tfidf,
    y_train
)

nb_pred = nb_model.predict(
    X_test_tfidf
)

# ==========================================
# SVM
# ==========================================

print("Training SVM...\n")

svm_model = LinearSVC(
    class_weight="balanced"
)

svm_model.fit(
    X_train_tfidf,
    y_train
)

svm_pred = svm_model.predict(
    X_test_tfidf
)

# ==========================================
# PRINT RESULTS
# ==========================================

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")

print(classification_report(
    y_test,
    lr_pred,
    zero_division=0
))

print("\n==============================")
print("NAIVE BAYES")
print("==============================")

print(classification_report(
    y_test,
    nb_pred,
    zero_division=0
))

print("\n==============================")
print("SVM")
print("==============================")

print(classification_report(
    y_test,
    svm_pred,
    zero_division=0
))

# ==========================================
# COMPARISON TABLE
# ==========================================

results = pd.DataFrame({

    "Model": [

        "Logistic Regression",

        "Naive Bayes",

        "SVM"
    ],

    "Accuracy": [

        accuracy_score(y_test, lr_pred),

        accuracy_score(y_test, nb_pred),

        accuracy_score(y_test, svm_pred)
    ],

    "F1 Score": [

        f1_score(
            y_test,
            lr_pred,
            average="weighted"
        ),

        f1_score(
            y_test,
            nb_pred,
            average="weighted"
        ),

        f1_score(
            y_test,
            svm_pred,
            average="weighted"
        )
    ]
})

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(results)

# ==========================================
# SAVE RESULTS
# ==========================================

results.to_csv(

    "results/baseline_results.csv",

    index=False
)

print("\nResults Saved Successfully")

# ==========================================
# SAVE MODELS
# ==========================================

joblib.dump(

    lr_model,

    "models/baseline/logistic_regression.pkl"
)

joblib.dump(

    nb_model,

    "models/baseline/naive_bayes.pkl"
)

joblib.dump(

    svm_model,

    "models/baseline/svm.pkl"
)

joblib.dump(

    vectorizer,

    "models/baseline/tfidf_vectorizer.pkl"
)

print("\nModels Saved Successfully")
