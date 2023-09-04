import logging
import os
from typing import Dict, Any

from haystack import Pipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PromptNode, PromptTemplate, TopPSampler
from haystack.nodes.ranker import LostInTheMiddleRanker
from haystack.nodes.retriever.web import WebRetriever

search_key = os.environ.get("SERPERDEV_API_KEY")
if not search_key:
    raise ValueError("Please set the SERPERDEV_API_KEY environment variable")

models_config: Dict[str, Any] = {
    "openai": {"api_key": os.environ.get("OPENAI_API_KEY"), "model_name": "gpt-3.5-turbo"},
    "anthropic": {"api_key": os.environ.get("ANTHROPIC_API_KEY"), "model_name": "claude-instant-1"},
    "hf": {"api_key": os.environ.get("HF_API_KEY"), "model_name": "tiiuae/falcon-7b-instruct"},
}
prompt_text = """
Synthesize a comprehensive answer from the provided paragraphs and the given question.\n
Focus on the question and avoid unnecessary information in your answer.\n
\n\n Paragraphs: {join(documents)} \n\n Question: {query} \n\n Answer:
"""

stream = True
model: Dict[str, str] = models_config["openai"]
prompt_node = PromptNode(
    model["model_name"],
    default_prompt_template=PromptTemplate(prompt_text),
    api_key=model["api_key"],
    max_length=768,
    model_kwargs={"stream": stream},
)

web_retriever = WebRetriever(
    api_key=search_key,
    allowed_domains=["haystack.deepset.ai"],
    top_search_results=10,
    mode="preprocessed_documents",
    top_k=50,
    cache_document_store=InMemoryDocumentStore(),
)

pipeline = Pipeline()
pipeline.add_node(component=web_retriever, name="Retriever", inputs=["Query"])
pipeline.add_node(component=TopPSampler(top_p=0.90), name="Sampler", inputs=["Retriever"])
pipeline.add_node(component=LostInTheMiddleRanker(1024), name="LostInTheMiddleRanker", inputs=["Sampler"])
pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["LostInTheMiddleRanker"])

logging.disable(logging.CRITICAL)

test = False
questions = [
    "What are the main benefits of using pipelines in Haystack?",
    "Are there any ready-made pipelines available and why should I use them?",
]

print(f"Running pipeline with {model['model_name']}\n")

if test:
    for question in questions:
        if stream:
            print("Answer:")
        response = pipeline.run(query=question)
        if not stream:
            print(f"Answer: {response['results'][0]}")
else:
    while True:
        user_input = input("\nAsk question (type 'exit' or 'quit' to quit): ")
        if user_input.lower() == "exit" or user_input.lower() == "quit":
            break
        if stream:
            print("Answer:")
        response = pipeline.run(query=user_input)
        if not stream:
            print(f"Answer: {response['results'][0]}")
