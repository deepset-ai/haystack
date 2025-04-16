import glob

from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import Document
from haystack.dataclasses.chat_message import ChatMessage, ImageContent
from haystack.document_stores.in_memory import InMemoryDocumentStore

docs = []

for image_path in glob.glob("examples/arxiv_images/*.png"):
    text = "image from '" + image_path.split("/")[-1].replace(".png", "").replace("_", " ") + "' paper"
    docs.append(Document(content=text, meta={"image_path": image_path}))

document_store = InMemoryDocumentStore()
document_store.write_documents(docs)

retriever = InMemoryBM25Retriever(document_store=document_store)

query = "What the image from the spectrum paper tries to show?"

doc = retriever.run(query=query, top_k=1)["documents"][0]

print(f"retrieved document: {doc}")

image_content = ImageContent.from_file_path(doc.meta["image_path"], detail="auto")

message = ChatMessage.from_user(content_parts=[query, image_content])

generator = OpenAIChatGenerator(model="gpt-4o-mini")

response = generator.run(messages=[message])

print(f"response: {response['replies'][0].text}")
# The image presents a bar chart comparing the performance of different Mistral-7b models across various evaluation
# metrics. Each bar represents the scores achieved by different training methods:
# - **qLora (orange)**
# - **Spectrum (25% and 50%, light and dark pink)**
# - **FFT (red)**
# The scores for each method are labeled on top of the bars, showing how well each technique performs on specific
# metrics: `arc`, `gsm8k`, `hellaswag`, `mmlu`, `truthfulqa`, and `winogrande`. The chart highlights differences in
# performance, indicating which method yields the best results for the evaluated models.
