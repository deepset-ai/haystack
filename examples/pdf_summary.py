import os
from haystack.nodes import PromptNode, LinkContentFetcher, PromptTemplate
from haystack import Pipeline

openai_key = os.environ.get("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

alt_user_agents = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15"
]

retriever = LinkContentFetcher(user_agents=alt_user_agents)
pt = PromptTemplate(
    "Given the content below, create a summary consisting of three sections: Objectives, "
    "Implementation and Learnings/Conclusions.\n"
    "Each section should have at least three bullet points. \n"
    "In the content below disregard References section.\n\n: {documents}"
)

prompt_node = PromptNode(
    "gpt-3.5-turbo-16k-0613",
    api_key=openai_key,
    max_length=512,
    default_prompt_template=pt,
    model_kwargs={"stream": True},
)

pipeline = Pipeline()
pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])

research_papers = ["https://arxiv.org/pdf/2307.03172.pdf", "https://arxiv.org/pdf/1706.03762.pdf"]

for research_paper in research_papers:
    print(f"Research paper summary: {research_paper}")
    pipeline.run(research_paper)
    print("\n\n\n")
