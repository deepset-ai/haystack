import os

from haystack import Pipeline
from haystack.nodes import PromptNode, PromptTemplate, Shaper
from haystack.nodes.retriever.web import WebRetriever

search_key = os.environ.get("SERPERDEV_API_KEY")
if not search_key:
    raise ValueError("Please set the SERPERDEV_API_KEY environment variable")

openai_key = os.environ.get("OPENAI_API_KEY")
if not search_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")


web_retriever = WebRetriever(api_key=search_key)

prompt_text = """
Synthesize a comprehensive answer from the following most relevant paragraphs and the given question.
Provide a clear and concise response that summarizes the key points and information presented in the paragraphs.
Your answer should be in your own words and be no longer than 50 words.
\n\n Paragraphs: $documents \n\n Question: $query \n\n Answer:
"""

prompt_node = PromptNode(
    "text-davinci-003",
    default_prompt_template=PromptTemplate("lfqa", prompt_text=prompt_text),
    api_key=openai_key,
    max_length=256,
)
# Shaper helps us concatenate most relevant docs that we want to use as the context for the generated answer
shaper = Shaper(func="join_documents", inputs={"documents": "documents"}, outputs=["documents"])

pipeline = Pipeline()
pipeline.add_node(component=web_retriever, name="web_retriever", inputs=["Query"])
pipeline.add_node(component=shaper, name="shaper", inputs=["web_retriever"])
pipeline.add_node(component=prompt_node, name="prompt_node", inputs=["shaper"])

# Long-Form QA requiring multiple context paragraphs for the synthesis of an elaborate generative answer
questions = [
    "What are the advantages of EmbeddingRetriever in Haystack?",
    "What are the advantages of PromptNode in Haystack?",
    "What PromptNode invocation layers are available in Haystack?",
]

for q in questions:
    print(f"Question: {q}")
    response = pipeline.run(query=q)
    print(f"Answer: {response['results'][0]}")
