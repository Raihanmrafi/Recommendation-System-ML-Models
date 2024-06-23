# Recommendation-System-ML-Models
## 1. Content-Based Filtering with AutoEncoder
### Overview 
This script implements a content-based filtering approach using a simple AutoEncoder model to recommend products based on their features.
### Requirements 
Python 3.12
Libraries: pandas, tensorflow, scikit-learn, matplotlib
### Installation 
pip install pandas tensorflow scikit-learn matplotlib
### Description 
1. Loads product data from product.csv and selects relevant columns (id, name, order_click, history_view_product, min_order).
2. Normalizes numerical features (order_click, history_view_product, min_order) using StandardScaler.
3. Constructs an AutoEncoder model with TensorFlow/Keras to learn latent representations of products.
4. Trains the model to reconstruct input features and evaluates performance using Mean Squared Error (MSE).
5. Visualizes training and validation losses using matplotlib.
6. Implements cosine similarity to generate product recommendations based on learned embeddings.

## 2. Semantic Search with Sentence Transformers
### Overview
This script utilizes Sentence Transformers to perform semantic search and recommend products based on text similarity.
### Requirements 
Python 3.12 
Libraries: pandas, sentence-transformers
### Description 
1. Loads product data from product.csv.
2. Utilizes Sentence Transformers (paraphrase-xlm-r-multilingual-v1) to encode product names into semantic embeddings.
3. Implements a function to recommend products similar to a given product name based on embedding similarity.
4. Combines results from embedding similarity and substring matching for robust recommendations.
