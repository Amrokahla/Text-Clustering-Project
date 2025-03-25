import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("punkt")
nltk.download("stopwords")
def load_data(filepath):
    """Load dataset from CSV."""
    return pd.read_csv(filepath)

def clean_text(text: str) -> str:
    """Cleans and preprocesses text by removing URLs, mentions, punctuation, stopwords, and applying stemming."""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)

def preprocess_data(df, text_column="text"):
    """Applies text cleaning to an entire dataset."""
    df[text_column] = df[text_column].astype(str).apply(clean_text)
    return df
