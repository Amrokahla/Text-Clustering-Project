import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import umap.umap_ as umap

def get_sbert_embeddings(texts, model_name="nli-roberta-base-v2", n_components=10):
    """Generate SBERT embeddings with UMAP dimensionality reduction."""
    sbert_model = SentenceTransformer(model_name)
    embeddings = sbert_model.encode(texts, convert_to_numpy=True)

    umap_reducer = umap.UMAP(n_components=n_components, random_state=42)
    reduced_embeddings = umap_reducer.fit_transform(embeddings)
    
    return reduced_embeddings

def get_tfidf_features(texts, max_features=5000):
    """Convert text into TF-IDF vectors."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    return vectorizer.fit_transform(texts), vectorizer
    
def get_word2vec_embeddings(texts, model_name='glove-wiki-gigaword-100', dimension=100):
    """Generate Word2Vec embeddings for a list of texts (Now using GloVe)."""
    word_vectors = api.load(model_name)
    
    def text_to_embedding(text):
        words = text.split()
        valid_words = [word for word in words if word in word_vectors.key_to_index]
        
        if not valid_words:
            return np.zeros(dimension)
        
        word_embeddings = [word_vectors[word] for word in valid_words]
        return np.mean(word_embeddings, axis=0)
    
    embeddings = np.array([text_to_embedding(text) for text in texts])
    return embeddings

