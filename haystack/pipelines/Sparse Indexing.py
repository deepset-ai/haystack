# Create a dictionary-based sparse index
sparse_index = {}

# Index a document
document_id = 1
document = "Paris is the capital of France."
tokens = document.split()  # Tokenize the document

for token in tokens:
    if token not in sparse_index:
        sparse_index[token] = [document_id]
    else:
        sparse_index[token].append(document_id)

# Search for documents containing a specific term
term = "capital"
if term in sparse_index:
    matching_document_ids = sparse_index[term]
    print(f"Documents containing '{term}': {matching_document_ids}")
