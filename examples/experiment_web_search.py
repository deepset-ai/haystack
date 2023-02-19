import os
import re
from typing import Dict, Any, List

from haystack import Document
from haystack.nodes import PromptNode, PromptTemplate
from haystack.nodes.search_engine import WebSearch, NeuralWebSearch

search_key = os.environ.get("SERPERDEV_API_KEY")
if not search_key:
    raise ValueError("Please set the SERPERDEV_API_KEY environment variable")

ws = WebSearch(api_key=search_key, top_p=0.95, top_k=5, strict_top_k=False)

pn = PromptNode("text-davinci-003", api_key=os.environ.get("OPENAI_API_KEY"), max_length=256)

prompt_text = """
Answer the following question using the paragraphs below as sources. An answer should be short, a few words at most.
Provide the answer as the last generated line starting with Answer:

Paragraphs: $paragraphs

Question: $question

Instructions: Consider all the paragraphs above and their corresponding scores to generate the answer. While a single
paragraph may have a high score, it's important to consider all paragraphs for the same answer candidate to answer
accurately.

Let's think step-by-step, we have the following distinct answer possibilities:

"""
pt = PromptTemplate("neural_web_search", prompt_text=prompt_text)


def prepare_prompt_params(results: List[Document], invocation_context: Dict[str, Any]):
    paragraphs = "\n".join([f"-[{doc.meta['score']}] {doc.content}" for doc in results])
    return {"paragraphs": paragraphs, "question": invocation_context.get("query")}


def prepare_final_answer(prompt_node_response: str):
    answer_text = "No answer"
    answer_regex = r"Answer:\s*(.*)"
    answer_match = re.search(answer_regex, prompt_node_response)

    if answer_match:
        answer_text = answer_match.group(1).strip()
    return answer_text


ns = NeuralWebSearch(
    websearch=ws,
    prompt_node=pn,
    prompt_template=pt,
    prepare_template_params_fn=prepare_prompt_params,
    extract_final_answer_fn=prepare_final_answer,
)

questions = [
    "Who won the 1971 San Francisco mayoral election?",
    "Where was Jeremy McKinnon born?",
    "What river is near Dundalk, Ireland?",
    "Who is Kyle Moran?",
    "What party does Joseph Alioto belong to?",
    "When was the Democratic Party founded?",
    "Who is Olivia Wilde's boyfriend?",
]

for q in questions:
    response, _ = ns.run(q)
    print(f"{q} - {response['output']}")
