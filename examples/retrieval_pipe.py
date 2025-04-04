import glob
from pathlib import Path

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
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

pipeline = Pipeline()

chat_template = [
    ChatMessage.from_user(
        content_parts=["{{query}}", ImageContent(base64_image="{{documents[0] | document_to_image}}")]
    )
]
pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
pipeline.add_component("prompt_builder", ChatPromptBuilder(template=chat_template))
pipeline.add_component("generator", OpenAIChatGenerator(model="gpt-4o-mini"))

pipeline.connect("retriever.documents", "prompt_builder.documents")
pipeline.connect("prompt_builder.prompt", "generator.messages")

query = "What the image from the TextGrad paper tries to show?"

print(pipeline.run(data={"query": query}))
