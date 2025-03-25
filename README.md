# Text Clustering & Topic Modeling

This project applies **K-Means clustering** and **LDA topic modeling** to group similar text data, using advanced embedding techniques like **Sentence-BERT**.

## Features

- **Preprocessing**: Cleans text (removes URLs, punctuation, stopwords, applies stemming)
- **Embeddings**: 
  - Sentence-BERT (SBERT)
  - TF-IDF Vectorization
  - Word2Vec embeddings
- **Clustering**: 
  - K-Means for document grouping
  - Latent Dirichlet Allocation (LDA) for topic modeling
- **Evaluation**: Silhouette Score for clustering quality
- **Visualization**: PCA-based cluster plotting with dimensionality reduction

## Installation & Setup

It's a better practice to create a __virtual environment__ before using the project

### 1. Clone the Repository

```bash
git clone https://github.com/Amrokahla/Text-Clustering-Project.git
cd Text-Clustering-Project
```

### 2. Install Dependencies

Make sure you have Python 3.10+ installed. Then, run:

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Resources (One-time setup)

```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
```

## Dynamic Usage

The `main.py` script supports multiple embedding and clustering techniques:

### Data Download
The data being used is [People Wiki Data](https://drive.google.com/file/d/1_oqcPdcdM2Rf0F5VasNgxqmZBP-c50Hd/view?usp=sharing)

### Embedding Techniques
- `sbert`: Sentence-BERT embeddings (default)
- `tfidf`: TF-IDF Vectorization
- `word2vec`: Word2Vec embeddings `word2vec-google-news-300`

### Clustering Methods
- `kmeans`: K-Means Clustering
- `lda`: Latent Dirichlet Allocation Topic Modeling

### Example Commands

1. Default (SBERT embeddings with K-Means):
```bash
python src/main.py
```

2. TF-IDF with K-Means and 5 clusters:
```bash
python src/main.py --embedding tfidf --method kmeans --n-clusters 5
```

3. SBERT with LDA and 7 topics:
```bash
python src/main.py --method lda --n-topics 7
```

4. Word2Vec with K-Means:
```bash
python src/main.py --embedding word2vec --method kmeans
```

### Customization Options
- `--input-file`: Specify a custom input CSV
- `--embedding`: Choose embedding technique
- `--method`: Select clustering method
- `--n-clusters`: Set number of K-Means clusters
- `--n-topics`: Set number of LDA topics
- `--max-features`: Limit feature dimensions

## Project Structure

```
├── data/                  # Datasets (raw and preprocessed)
├── src/
│   ├── preprocessing.py    # Data cleaning & preprocessing
│   ├── feature_extraction.py  # Vectorization & embedding
│   ├── clustering.py       # Clustering models
│   ├── evaluation.py       # Clustering metrics
│   ├── visualization.py    # Result visualization
│   ├── main.py             # Main script for execution
├── notebooks/              # Jupyter notebooks for EDA
├── results/                # Cluster results & visuals
├── requirements.txt        # Dependencies
├── README.md               # Documentation
```

## Results

After running, you will see:
-  Cluster Assignments
-  Topic Distributions
-  **Visualizations of K-Means & LDA Clusters**

## Notes & Next Steps

- Experiment with different **SBERT models** (`all-MiniLM-L6-v2`, `mpnet-base-v2`)
- Try varying number of **clusters** or **topics**
- Fine-tune embedding and clustering parameters

## Contact & Contribution

Feel free to **contribute, suggest improvements, or report issues**.
