from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def get_sbert_embeddings(texts, model_name="nli-roberta-base-v2"):
    """Generate SBERT embeddings for text data."""
    model = SentenceTransformer(model_name)
    return model.encode(texts, convert_to_numpy=True)

def get_tfidf_features(texts, max_features=5000):
    """Convert text into TF-IDF vectors."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    return vectorizer.fit_transform(texts), vectorizer

def get_lda_features(texts, max_features=5000):
    """Convert text into word count vectors for LDA."""
    vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
    return vectorizer.fit_transform(texts), vectorizer

def get_bow_features(texts, max_features=5000):
    """Convert text into word count vectors."""
    vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
    return vectorizer.fit_transform(texts), vectorizer