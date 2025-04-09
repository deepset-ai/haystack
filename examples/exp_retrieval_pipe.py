import glob

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters.image import ImageToDocument
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses.chat_message import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore

document_store = InMemoryDocumentStore()
indexing_pipeline = Pipeline()
paths = glob.glob("arxiv_images/*.png")
texts = [
    "image from '" + image_path.split("/")[-1].replace(".png", "").replace("_", " ") + "' paper" for image_path in paths
]

indexing_pipeline.add_component("image_to_document", ImageToDocument())
indexing_pipeline.add_component("document_writer", DocumentWriter(document_store=document_store))
indexing_pipeline.connect("image_to_document.documents", "document_writer.documents")
indexing_pipeline.run(data={"sources": paths, "texts": texts})

query = "What the image from the Lora vs Full Fine-tuning paper tries to show? Be short."

rag_pipeline = Pipeline()

chat_template = [
    ChatMessage.from_user(content_parts=["{{query | as_text_content}}", "{{documents[0] | as_image_content}}"])
]
rag_pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store, top_k=1))
rag_pipeline.add_component("prompt_builder", ChatPromptBuilder(template=chat_template))
rag_pipeline.add_component("generator", OpenAIChatGenerator(model="gpt-4o-mini"))

rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "generator.messages")


print(rag_pipeline.run({"query": query}))

# {'generator': {'replies': [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text='
# The image from the LoRA vs. Full Fine-tuning paper illustrates the concept of "intruder dimensions" in model
# fine-tuning. \n\n1. **Part (a)** shows how LoRA adds low-rank updates (matrix \\(B\\)) to pre-trained weights
# (\\(W_0\\)), while full fine-tuning updates all weights.\n2. **Part (b)** compares the similarity of singular vectors
#  between LoRA and full fine-tuning, indicating that LoRA retains more of the original structure.\n3. **Part (c)**
# presents data on the cosine similarity of singular vectors, showing significant differences in how LoRA and full
# fine-tuning affect model behavior.\n\nOverall, it emphasizes how LoRA minimizes disruption to the original model by
# focusing on specific dimensions, whereas full fine-tuning leads to more substantial changes.')], _name=None, _meta={
# 'model': 'gpt-4o-mini-2024-07-18', 'index': 0, 'finish_reason': 'stop', 'usage': {'completion_tokens': 168,
# 'prompt_tokens': 2860, 'total_tokens': 3028, 'completion_tokens_details': CompletionTokensDetails(
# accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0),
# 'prompt_tokens_details': PromptTokensDetails(audio_tokens=0, cached_tokens=0)}})]}}
