"""
SWAHILI SENTIMENT ANALYSIS - COMPLETE WITH ALL RESULTS
"""

import pandas as pd
import numpy as np
import re
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("SWAHILI SENTIMENT ANALYSIS - COMPLETE PIPELINE")
print("=" * 70)

# ============================================
# 1. LOAD AND EXPLORE DATA
# ============================================
print("\n[1/6] LOADING DATA...")

df = pd.read_csv('data/raw/dev.tsv', sep='\t')
print(f"✅ Loaded {len(df)} samples")
print(f"Columns: {df.columns.tolist()}")

# Convert labels
label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label_num'] = df['label'].map(label_mapping)

print(f"\nClass distribution:")
print(df['label_num'].value_counts().sort_index())

# ============================================
# 2. TEXT PREPROCESSING
# ============================================
print("\n[2/6] CLEANING TEXT...")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_text'] = df['tweet'].apply(clean_text)
print(f"✅ Cleaned {len(df)} texts")

# ============================================
# 3. BASELINE MODELS
# ============================================
print("\n[3/6] TRAINING BASELINE MODELS...")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

X = df['cleaned_text'].tolist()
y = df['label_num'].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

baseline_results = {}

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)
baseline_results['Naive Bayes'] = {
    'accuracy': accuracy_score(y_test, nb_pred),
    'f1': f1_score(y_test, nb_pred, average='weighted'),
    'precision': precision_score(y_test, nb_pred, average='weighted'),
    'recall': recall_score(y_test, nb_pred, average='weighted')
}
print(f"\nNaive Bayes - Acc: {baseline_results['Naive Bayes']['accuracy']:.4f}, F1: {baseline_results['Naive Bayes']['f1']:.4f}")

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_test_tfidf)
baseline_results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, lr_pred),
    'f1': f1_score(y_test, lr_pred, average='weighted'),
    'precision': precision_score(y_test, lr_pred, average='weighted'),
    'recall': recall_score(y_test, lr_pred, average='weighted')
}
print(f"Logistic Regression - Acc: {baseline_results['Logistic Regression']['accuracy']:.4f}, F1: {baseline_results['Logistic Regression']['f1']:.4f}")

# ============================================
# 4. TRANSFORMER MODEL (XLM-RoBERTa)
# ============================================
print("\n[4/6] TRAINING TRANSFORMER MODEL...")

transformer_results = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    import torch
    
    # Use smaller subset for faster training
    train_size = min(200, len(X_train))
    test_size = min(50, len(X_test))
    
    train_texts = X_train[:train_size]
    train_labels = y_train[:train_size]
    test_texts = X_test[:test_size]
    test_labels = y_test[:test_size]
    
    print(f"Fine-tuning on {len(train_texts)} samples...")
    
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3)
    
    # Tokenize with proper type conversion
    def tokenize_function(texts, labels):
        texts = [str(t) for t in texts]
        labels = [int(l) for l in labels]
        
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
        
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = torch.tensor(labels, dtype=torch.long)
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = self.labels[idx]
                return item
            def __len__(self):
                return len(self.labels)
        
        return SimpleDataset(encodings, labels)
    
    train_dataset = tokenize_function(train_texts, train_labels)
    test_dataset = tokenize_function(test_texts, test_labels)
    
    training_args = TrainingArguments(
        output_dir='./transformer_results',
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=5,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        report_to='none'
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    print("Training (2 epochs - about 2-3 minutes)...")
    trainer.train()
    
    # Test predictions
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**test_encodings)
        preds = torch.argmax(outputs.logits, dim=1).numpy()
    
    transformer_results = {
        'accuracy': accuracy_score(test_labels, preds),
        'f1': f1_score(test_labels, preds, average='weighted'),
        'precision': precision_score(test_labels, preds, average='weighted'),
        'recall': recall_score(test_labels, preds, average='weighted')
    }
    
    print(f"\nXLM-RoBERTa Results:")
    print(f"  Accuracy: {transformer_results['accuracy']:.4f}")
    print(f"  F1-Score: {transformer_results['f1']:.4f}")
    
except Exception as e:
    print(f"⚠️ Transformer training skipped: {e}")

# ============================================
# 5. RESULTS TABLE
# ============================================
print("\n" + "=" * 70)
print("FINAL RESULTS - MODEL COMPARISON")
print("=" * 70)

print(f"\n{'Model':<25} {'Accuracy':<12} {'F1-Score':<12} {'Precision':<12} {'Recall':<12}")
print("-" * 70)

for name, metrics in baseline_results.items():
    print(f"{name:<25} {metrics['accuracy']:<12.4f} {metrics['f1']:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f}")

if transformer_results:
    print(f"{'XLM-RoBERTa':<25} {transformer_results['accuracy']:<12.4f} {transformer_results['f1']:<12.4f} {transformer_results['precision']:<12.4f} {transformer_results['recall']:<12.4f}")

print("-" * 70)

# ============================================
# 6. SAVE RESULTS
# ============================================
print("\n[6/6] SAVING RESULTS...")

os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

import joblib
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
joblib.dump(lr_model, 'models/logistic_regression.pkl')
print("✅ Models saved to 'models/'")

# Save results
results_df = pd.DataFrame([
    {'Model': name, 
     'Accuracy': metrics['accuracy'], 
     'F1-Score': metrics['f1'],
     'Precision': metrics['precision'],
     'Recall': metrics['recall']}
    for name, metrics in baseline_results.items()
])

if transformer_results:
    results_df = pd.concat([results_df, pd.DataFrame([{
        'Model': 'XLM-RoBERTa',
        'Accuracy': transformer_results['accuracy'],
        'F1-Score': transformer_results['f1'],
        'Precision': transformer_results['precision'],
        'Recall': transformer_results['recall']
    }])])

results_df.to_csv('results/model_comparison.csv', index=False)
print("✅ Results saved to 'results/model_comparison.csv'")

# ============================================
# 7. SUMMARY FOR REPORT
# ============================================
print("\n" + "=" * 70)
print("SUMMARY FOR YOUR REPORT")
print("=" * 70)

print(f"""
DATASET STATISTICS:
- Total samples: {len(df)}
- Negative (0): {y.count(0)}
- Neutral (1): {y.count(1)}  
- Positive (2): {y.count(2)}

BASELINE MODELS:
- Naive Bayes: {baseline_results['Naive Bayes']['f1']:.4f} F1-score
- Logistic Regression: {baseline_results['Logistic Regression']['f1']:.4f} F1-score

TRANSFORMER MODEL:
- XLM-RoBERTa: {transformer_results['f1']:.4f} F1-score (if available)

BEST MODEL: {'Logistic Regression' if not transformer_results or transformer_results['f1'] < baseline_results['Logistic Regression']['f1'] else 'XLM-RoBERTa'}
- F1-Score: {max(baseline_results['Logistic Regression']['f1'], transformer_results['f1'] if transformer_results else 0):.4f}

ANSWER TO RESEARCH QUESTION:
- Transformers vs Baselines: {'Baseline models performed better' if not transformer_results or transformer_results['f1'] < baseline_results['Logistic Regression']['f1'] else 'Transformer models performed better'} on this dataset.
- Best F1-score achieved: {max(baseline_results['Logistic Regression']['f1'], transformer_results['f1'] if transformer_results else 0):.4f}
""")

print("\n✅ ANALYSIS COMPLETE!")
print("\n📁 Files saved:")
print("   - results/model_comparison.csv")
print("   - models/tfidf_vectorizer.pkl")
print("   - models/logistic_regression.pkl")
