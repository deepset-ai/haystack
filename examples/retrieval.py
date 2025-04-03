import base64
import glob
from pathlib import Path

from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ByteStream, Document
from haystack.dataclasses.chat_message import ChatMessage, ImageContent
from haystack.document_stores.in_memory import InMemoryDocumentStore

docs = []

for image_path in glob.glob("arxiv_images/*.png"):
    text = "image from '" + image_path.split("/")[-1].replace(".png", "").replace("_", " ") + "' paper"
    docs.append(Document(content=text, blob=ByteStream.from_file_path(Path(image_path))))

document_store = InMemoryDocumentStore()
document_store.write_documents(docs)

retriever = InMemoryBM25Retriever(document_store=document_store)

query = "What the image from the spectrum paper tries to show?"

result = retriever.run(query=query, top_k=1)["documents"][0]

print(f"retrieved document: {result}")

message = ChatMessage.from_user(
    content_parts=[query, ImageContent(base64_image=base64.b64encode(result.blob.data).decode("utf-8"))]
)

generator = OpenAIChatGenerator(model="gpt-4o-mini")

response = generator.run(messages=[message])

print(f"response: {response['replies'][0].text}")
