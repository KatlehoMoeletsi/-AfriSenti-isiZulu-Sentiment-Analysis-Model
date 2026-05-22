import pandas as pd
import nlpaug.augmenter.word as naw
from googletrans import Translator

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/processed/cleaned_data.csv"
)

translator = Translator()

# ==========================================
# SYNONYM AUGMENTATION
# ==========================================

aug = naw.SynonymAug(aug_src='wordnet')

def synonym_augment(text):

    try:
        return aug.augment(text)

    except:
        return text

# ==========================================
# BACK TRANSLATION
# ==========================================

def back_translate(text):

    try:

        english = translator.translate(
            text,
            src='sw',
            dest='en'
        ).text

        swahili = translator.translate(
            english,
            src='en',
            dest='sw'
        ).text

        return swahili

    except:
        return text

# ==========================================
# APPLY AUGMENTATION
# ==========================================

df['synonym_augmented'] = df['clean_text'].apply(
    synonym_augment
)

df['back_translated'] = df['clean_text'].apply(
    back_translate
)

# ==========================================
# SAVE
# ==========================================

df.to_csv(
    "data/processed/augmented_data.csv",
    index=False
)

print("Augmentation completed.")
