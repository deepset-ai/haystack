import logging
import os

from haystack import Pipeline
from haystack.nodes import PromptNode, PromptTemplate, TopPSampler, DocumentMerger
from haystack.nodes.ranker.diversity import DiversityRanker
from haystack.nodes.retriever.web import WebRetriever

search_key = os.environ.get("SERPERDEV_API_KEY")
if not search_key:
    raise ValueError("Please set the SERPERDEV_API_KEY environment variable")

openai_key = os.environ.get("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

prompt_text = """
Synthesize a comprehensive answer from the following most relevant paragraphs and the given question.
Provide a clear and concise response that summarizes the key points and information presented in the paragraphs.
Your answer should be in your own words and be no longer than 50 words.
\n\n Paragraphs: {documents} \n\n Question: {query} \n\n Answer:
"""

prompt_node = PromptNode(
    "gpt-3.5-turbo", default_prompt_template=PromptTemplate(prompt_text), api_key=openai_key, max_length=256
)

web_retriever = WebRetriever(api_key=search_key, top_search_results=10, mode="preprocessed_documents", top_k=25)

sampler = TopPSampler(top_p=0.95)
ranker = DiversityRanker()
merger = DocumentMerger(separator="\n\n")

pipeline = Pipeline()
pipeline.add_node(component=web_retriever, name="Retriever", inputs=["Query"])
pipeline.add_node(component=sampler, name="Sampler", inputs=["Retriever"])
pipeline.add_node(component=ranker, name="Ranker", inputs=["Sampler"])
pipeline.add_node(component=merger, name="Merger", inputs=["Ranker"])
pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Merger"])

logger = logging.getLogger("boilerpy3")
logger.setLevel(logging.CRITICAL)

questions = ["What are the reasons for long-standing animosities between Russia and Poland?"]

for q in questions:
    print(f"Question: {q}")
    response = pipeline.run(query=q)
    print(f"Answer: {response['results'][0]}")
