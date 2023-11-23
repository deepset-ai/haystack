import logging
import os
from typing import Dict, Any

from haystack import Pipeline
from haystack.nodes import PromptNode, PromptTemplate, TopPSampler
from haystack.nodes.ranker import DiversityRanker, LostInTheMiddleRanker
from haystack.nodes.retriever import WebRetriever

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
Answer in full sentences and paragraphs, don't use bullet points or lists.\n
If the answer includes multiple chronological events, order them chronologically.\n
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

web_retriever = WebRetriever(api_key=search_key, top_search_results=5, mode="preprocessed_documents", top_k=50)

sampler = TopPSampler(top_p=0.97)
diversity_ranker = DiversityRanker()
litm_ranker = LostInTheMiddleRanker(word_count_threshold=1024)

pipeline = Pipeline()
pipeline.add_node(component=web_retriever, name="Retriever", inputs=["Query"])
pipeline.add_node(component=sampler, name="Sampler", inputs=["Retriever"])
pipeline.add_node(component=diversity_ranker, name="DiversityRanker", inputs=["Sampler"])
pipeline.add_node(component=litm_ranker, name="LostInTheMiddleRanker", inputs=["DiversityRanker"])
pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["LostInTheMiddleRanker"])

logging.disable(logging.CRITICAL)


questions = [
    "What are the primary causes and effects of climate change on global and local scales?",
    "What were the key events and influences that led to Renaissance; how did these developments shape modern Western culture?",
    "How have advances in technology in the 21st century affected job markets and economies around the world?",
    "How has the European Union influenced the political, economic, and social dynamics of Europe?",
]

print(f"\nRunning pipeline with {model['model_name']}\n")

for q in questions:
    print(f"\nQuestion: {q}")
    if stream:
        print("Answer:")
    response = pipeline.run(query=q)
    if not stream:
        print(f"Answer: {response['results'][0]}")
