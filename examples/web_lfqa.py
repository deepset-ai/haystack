import os

from haystack import Pipeline
from haystack.nodes import PromptNode, PromptTemplate, Shaper
from haystack.nodes.retriever.web import WebRetriever

search_key = os.environ.get("SERPAPI_API_KEY")
if not search_key:
    raise ValueError("Please set the SERPAPI_API_KEY environment variable")

openai_key = os.environ.get("OPENAI_API_KEY")
if not search_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")


web_retriever = WebRetriever(api_key=search_key, search_engine_provider="SerpAPI")

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

questions = [
    # "Who won the 1971 San Francisco mayoral election?",
    # "Where was Jeremy McKinnon born?",
    # "What river is near Dundalk, Ireland?",
    # "What party does Joseph Alioto belong to?",
    # "When was the Democratic Party founded?",
    # "Who is Olivia Wilde's boyfriend?",
    "What are the advantages of EmbeddingRetriever in Haystack?",
    "What are the advantages of PromptNode in Haystack?",
    "What PromptNode invocation layers are available in Haystack?",
]

for q in questions:
    response = pipeline.run(query=q)
    print(f"{q} - {response['results'][0]}")
