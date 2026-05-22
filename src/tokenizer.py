import sentencepiece as spm
import pandas as pd

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/processed/cleaned_data.csv"
)

# ==========================================
# SAVE TEXT FILE
# ==========================================

with open(
    "data/processed/cleaned_text.txt",
    "w",
    encoding="utf-8"
) as f:

    for text in df['clean_text']:
        f.write(text + "\n")

# ==========================================
# TRAIN TOKENIZER
# ==========================================

spm.SentencePieceTrainer.train(
    input='data/processed/cleaned_text.txt',
    model_prefix='models/swahili_sp',
    vocab_size=8000
)

print("SentencePiece model trained.")

# ==========================================
# LOAD TOKENIZER
# ==========================================

sp = spm.SentencePieceProcessor()

sp.load('models/swahili_sp.model')

# ==========================================
# TEST TOKENIZATION
# ==========================================

example = "ninapenda huduma hii"

tokens = sp.encode_as_pieces(example)

print(tokens)