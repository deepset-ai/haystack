import os
from haystack.nodes import PromptNode, LinkContentFetcher, PromptTemplate
from haystack import Pipeline

anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
if not anthropic_key:
    raise ValueError("Please set the ANTHROPIC_API_KEY environment variable")

retriever = LinkContentFetcher()
pt = PromptTemplate(
    "Given the paragraphs of the blog post, "
    "provide the main learnings and the final conclusion using short bullet points format."
    "\n\nParagraphs: {documents}"
)

prompt_node = PromptNode(
    "claude-instant-1", api_key=anthropic_key, max_length=512, default_prompt_template=pt, model_kwargs={"stream": True}
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
