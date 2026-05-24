# ==========================================
# IMPORTS
# ==========================================

import os
import joblib
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    classification_report,

    confusion_matrix
)

# ==========================================
# CREATE FOLDERS
# ==========================================

os.makedirs("results/graphs", exist_ok=True)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/processed/cleaned_data.csv"
)

print("\nDataset Loaded Successfully\n")

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

# ==========================================
# LOAD VECTORIZER
# ==========================================

vectorizer = joblib.load(
    "models/baseline/tfidf_vectorizer.pkl"
)

# ==========================================
# TRANSFORM TEST DATA
# ==========================================

X_test_tfidf = vectorizer.transform(
    X_test
)

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(
    "models/baseline/svm.pkl"
)

print("SVM Model Loaded Successfully\n")

# ==========================================
# PREDICTIONS
# ==========================================

predictions = model.predict(
    X_test_tfidf
)

# ==========================================
# METRICS
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted"
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted"
)

# ==========================================
# PRINT RESULTS
# ==========================================

print("\n==============================")
print("SVM MODEL EVALUATION")
print("==============================")

print(f"Accuracy  : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions
)

# ==========================================
# PLOT MATRIX
# ==========================================

plt.figure(figsize=(8,6))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues"
)

plt.title("SVM Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

# ==========================================
# SAVE FIGURE
# ==========================================

plt.savefig(

    "results/graphs/svm_confusion_matrix.png"
)

#plt.show()

print("\nConfusion Matrix Saved Successfully")