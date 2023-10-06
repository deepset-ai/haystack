import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Sample documents
documents = ["Paris is the capital of France.", "London is the capital of the UK.", "Berlin is in Germany."]

# Compute TF-IDF vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Search for documents using vector similarity
query = "capital of France"
query_vector = tfidf_vectorizer.transform([query])

# Calculate cosine similarity between the query vector and document vectors
cosine_similarities = np.dot(tfidf_matrix, query_vector.T).toarray().flatten()

# Sort and get the most relevant document(s)
sorted_indices = np.argsort(cosine_similarities)[::-1]
top_document_index = sorted_indices[0]
print(f"Most relevant document: {documents[top_document_index]}")
