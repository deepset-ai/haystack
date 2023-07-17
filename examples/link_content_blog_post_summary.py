import os
from haystack.nodes import PromptNode, LinkContentFetcher, PromptTemplate
from haystack import Pipeline

openai_key = os.environ.get("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

retriever = LinkContentFetcher()
pt = PromptTemplate(
    "Given the paragraphs of the blog post, "
    "provide the main learnings and the final conclusion using short bullet points format."
    "\n\nParagraphs: {documents}"
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

blog_posts = [
    "https://pythonspeed.com/articles/base-image-python-docker-images/",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
]

for blog_post in blog_posts:
    print(f"Blog post summary: {blog_post}")
    pipeline.run(blog_post)
    print("\n\n\n")
