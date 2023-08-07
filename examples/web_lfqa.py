import logging
import os

from haystack.nodes import PromptNode, PromptTemplate, TopPSampler
from haystack.nodes.retriever.web import WebRetriever
from haystack.pipelines import WebQAPipeline

search_key = os.environ.get("SERPERDEV_API_KEY")
if not search_key:
    raise ValueError("Please set the SERPERDEV_API_KEY environment variable")

openai_key = os.environ.get("OPENAI_API_KEY")
if not search_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

prompt_text = """
Synthesize a comprehensive answer from the following most relevant paragraphs and the given question.
Provide a clear and concise response that summarizes the key points and information presented in the paragraphs.
Your answer should be in your own words and be no longer than 50 words.
\n\n Paragraphs: {documents} \n\n Question: {query} \n\n Answer:
"""

prompt_node = PromptNode(
    "text-davinci-003", default_prompt_template=PromptTemplate(prompt_text), api_key=openai_key, max_length=256
)

web_retriever = WebRetriever(api_key=search_key, top_search_results=5, mode="preprocessed_documents", top_k=30)
pipeline = WebQAPipeline(retriever=web_retriever, prompt_node=prompt_node, sampler=TopPSampler(top_p=0.8))

# Long-Form QA requiring multiple context paragraphs for the synthesis of an elaborate generative answer
questions = [
    "What are the advantages of EmbeddingRetriever in Haystack?",
    "What are the advantages of PromptNode in Haystack?",
    "What PromptModelInvocationLayer implementations are available in Haystack?",
]

# Avoid all failed html parsing logs
logger = logging.getLogger("haystack.nodes.retriever.link_content")
logger.setLevel(logging.CRITICAL)
logger = logging.getLogger("boilerpy3")
logger.setLevel(logging.CRITICAL)


for q in questions:
    print(f"Question: {q}")
    response = pipeline.run(query=q)
    print(f"Answer: {response['results'][0]}")
