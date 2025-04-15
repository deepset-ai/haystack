import glob
from typing import List

from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import OutputAdapter
from haystack.components.converters.image import FileToImageContent
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses.chat_message import ChatMessage, ImageContent
from haystack.document_stores.in_memory import InMemoryDocumentStore

document_store = InMemoryDocumentStore()

paths = glob.glob("examples/arxiv_images/*.png")
texts = [
    "image from '" + image_path.split("/")[-1].replace(".png", "").replace("_", " ") + "' paper" for image_path in paths
]

docs = []
for path in paths:
    text = "image from '" + path.split("/")[-1].replace(".png", "").replace("_", " ") + "' paper"
    meta = {"image_path": path}
    docs.append(Document(content=text, meta=meta))

document_store.write_documents(docs)
print(document_store.filter_documents())


rag_pipeline = Pipeline()

chat_template = """
{% message role="user" %}
    {{query}}
    {% for image_content in image_contents %}
        {{image_content | for_template}}
    {% endfor %}
{% endmessage %}
"""

output_adapter_template = """
{%- set paths = [] -%}
{% for document in documents %}
    {%- set _ = paths.append(document.meta.image_path) -%}
{% endfor %}
{{paths}}
"""

rag_pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store, top_k=1))
rag_pipeline.add_component("output_adapter", OutputAdapter(template=output_adapter_template, output_type=List[str]))
rag_pipeline.add_component("image_converter", FileToImageContent(detail="auto"))
rag_pipeline.add_component("prompt_builder", ChatPromptBuilder(template=chat_template))
rag_pipeline.add_component("generator", OpenAIChatGenerator(model="gpt-4o-mini"))

rag_pipeline.connect("retriever.documents", "output_adapter.documents")
rag_pipeline.connect("output_adapter.output", "image_converter.sources")
rag_pipeline.connect("image_converter.image_contents", "prompt_builder.image_contents")
rag_pipeline.connect("prompt_builder.prompt", "generator.messages")

query = "What the image from the Lora vs Full Fine-tuning paper tries to show? Be short."

response = rag_pipeline.run(data={"query": query})["generator"]["replies"][0].text
print(response)

# The image compares two approaches to fine-tuning neural networks: LoRA (Low-Rank Adaptation) and full fine-tuning.

# 1. **Panel (a)** illustrates how LoRA introduces low-rank adjustments (represented by matrices \(A\) and \(B\)) to
# the pre-trained weights \(W_0\), while full fine-tuning modifies the weights directly with \(\Delta W\).
