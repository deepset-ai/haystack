import logging
from typing import Optional

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.pipeline_utils import build_rag_pipeline, build_indexing_pipeline
from haystack.pipeline_utils.indexing import download_files

# Set up logging
logging.basicConfig(level=logging.INFO)  # set to DEBUG for more info
logger = logging.getLogger(__name__)

# If working on your own fork of haystack
# Be sure to uninstall existing haystack: pip uninstall -y farm-haystack haystack-ai
# Then deploy changes with pip install -e '.[all]'
# sys.path.insert(0, '/home/hosermage/forked-projects/haystack')
# print(f"System path: {sys.path}")


# We are model agnostic :) In this getting started you can choose any OpenAI or Huggingface TGI generation model
generation_model = "gpt-3.5-turbo"
API_KEY = "sk-..."  # ADD YOUR KEY HERE
API_BASE_URL: Optional[str] = None
PROMPT_TEMPLATE: Optional[str] = None
SYSTEM_PROMPT: Optional[str] = None
# Adjust the below as needed.  If you want to connect to your own model, change the default
# prompt template or system prompt.
# API_BASE_URL = "http://192.168.1.100:1234/v1" #Uncomment if running local model with LM Studio or similar
# PROMPT_TEMPLATE = """
#                 Given these documents, answer the question.

#                 Documents:
#                 {% for doc in documents %}
#                     {{ doc.content }}
#                 {% endfor %}

#                 Question: {{question}}

#                 Answer:
#                 """
# SYSTEM_PROMPT = """
#                 You will be given a list of documents to answer a question.  Please reference ONLY
#                 these documents when formulating your answer.
#                 """

# We support many different databases. Here, we load a simple and lightweight in-memory database.
# Use cosine for e5-base-v2 https://discord.com/channels/993534733298450452/1141635410024468491/1141761753772998787
document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
logger.info("Document store configured: %s", document_store)
logger.info("Document store attributes:")
for attr, value in document_store.__dict__.items():
    logger.info("  %s: %s", attr, value)

# Download example files from web
sources = ["http://www.paulgraham.com/superlinear.html"]
logger.info("Downloading files from sources: %s", sources)
files = download_files(sources=sources)
logger.info("Downloaded files: %s", files)

# Pipelines are our main abstratcion.
# Here we create a pipeline that can index TXT and HTML. You can also use your own private files.
indexing_pipeline = build_indexing_pipeline(
    document_store=document_store,
    embedding_model="intfloat/e5-base-v2",
    supported_mime_types=["text/plain", "text/html"],  # "application/pdf"
)
indexing_pipeline.run(files=files)  # you can also supply files=[path_to_directory], which is searched recursively

logger.info("Building RAG Pipeline: %s", files)
# RAG pipeline with vector-based retriever + LLM
rag_pipeline = build_rag_pipeline(
    document_store=document_store,
    embedding_model="intfloat/e5-base-v2",
    generation_model=generation_model,
    llm_api_key=API_KEY,
    api_base_url=API_BASE_URL,
    prompt_template=PROMPT_TEMPLATE,
    system_prompt=SYSTEM_PROMPT,
)

# For details, like which documents were used to generate the answer, look into the result object
result = rag_pipeline.run(query="What are superlinear returns and why are they important?")
print(result.data)
