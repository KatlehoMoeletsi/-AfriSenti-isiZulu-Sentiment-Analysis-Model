import pandas as pd
import numpy as np
import re
import emoji
import nltk
from sklearn.model_selection import train_test_split

nltk.download('punkt')

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/raw/dev.tsv",
    sep="\t"
)

print(df.head())
print(df.columns)

# ==========================================
# CLEANING FUNCTION
# ==========================================

def clean_text(text):

    text = str(text).lower()

    # remove urls
    text = re.sub(r'http\S+', '', text)

    # remove mentions
    text = re.sub(r'@\w+', '', text)

    # remove hashtags
    text = re.sub(r'#\w+', '', text)

    # remove emojis
    text = emoji.replace_emoji(text, replace='')

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ==========================================
# APPLY CLEANING
# ==========================================

df['clean_text'] = df['text'].apply(clean_text)

# ==========================================
# LABEL ENCODING
# ==========================================

label_map = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}

df['label_encoded'] = df['label'].map(label_map)

# ==========================================
# REMOVE NULLS
# ==========================================

df.dropna(inplace=True)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'],
    df['label_encoded'],
    test_size=0.2,
    random_state=42,
    stratify=df['label_encoded']
)

# ==========================================
# SAVE FILES
# ==========================================

df.to_csv(
    "data/processed/cleaned_data.csv",
    index=False
)

X_train.to_csv(
    "data/processed/X_train.csv",
    index=False
)

X_test.to_csv(
    "data/processed/X_test.csv",
    index=False
)

y_train.to_csv(
    "data/processed/y_train.csv",
    index=False
)

y_test.to_csv(
    "data/processed/y_test.csv",
    index=False
)

print("Preprocessing completed.")
