from transformers import RagTokenizer, RagRetriever, RagModel
import torch

# Initialize the tokenizer, retriever, and model
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
retriever = RagRetriever.from_pretrained("facebook/rag-token-base", index_name="my_custom_index")
model = RagModel.from_pretrained("facebook/rag-token-base", retriever=retriever)

# Define a query
query = "Tell me about the Solar System."

# Encode the query and retrieve relevant documents
input_ids = tokenizer.encode(query, return_tensors="pt")
retrieved_docs = model.retrieve(input_ids)

# Generate a response based on retrieved documents
output = model.generate(input_ids, retrieved_docs=retrieved_docs)
response = tokenizer.decode(output[0], skip_special_tokens=True)

print("Response:", response)
