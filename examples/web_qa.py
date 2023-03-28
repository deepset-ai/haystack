import os
from haystack.nodes import PromptNode
from haystack.nodes.retriever.web import WebRetriever
from haystack.pipelines import WebQAPipeline

search_key = os.environ.get("SERPERDEV_API_KEY")
if not search_key:
    raise ValueError("Please set the SERPERDEV_API_KEY environment variable")

openai_key = os.environ.get("OPENAI_API_KEY")
if not search_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

prompt_node = PromptNode(
    "text-davinci-003",
    api_key=openai_key,
    max_length=256,
    default_prompt_template="question-answering-with-document-scores",
)
web_retriever = WebRetriever(api_key=search_key)
pipeline = WebQAPipeline(retriever=web_retriever, prompt_node=prompt_node)

questions = [
    "Who won the 1971 San Francisco mayoral election?",
    "Where was Jeremy McKinnon born?",
    "What river is near Dundalk, Ireland?",
    "Who is Kyle Moran?",
    "What party does Joseph Alioto belong to?",
    "When was the Democratic Party founded?",
    "Who is Olivia Wilde's boyfriend?",
]

for question in questions:
    response = pipeline.run(question)
    print(f"{question} - {response['answers'][0].answer}")
