import logging
import os

from haystack.utils import convert_files_to_docs
from haystack.utils import fetch_archive_from_http

logger = logging.getLogger(__name__)


def build_pipeline(provider, API_KEY, document_store):
    # Importing top-level causes a circular import
    from haystack.nodes import AnswerParser, PromptNode, PromptTemplate, BM25Retriever
    from haystack.pipelines import Pipeline

    provider = provider.lower()
    # A retriever selects the right documents when given a question.
    retriever = BM25Retriever(document_store=document_store, top_k=5)
    # Load prompt for doing retrieval augmented generation from https://prompthub.deepset.ai/?prompt=deepset%2Fquestion-answering-with-references
    question_answering_with_references = PromptTemplate(
        prompt="deepset/question-answering-with-references",
        output_parser=AnswerParser(reference_pattern=r"Document\[(\d+)\]"),
    )
    # Load the LLM model
    if provider == "anthropic":
        prompt_node = PromptNode(
            model_name_or_path="claude-2", api_key=API_KEY, default_prompt_template=question_answering_with_references
        )
    elif provider == "cohere":
        prompt_node = PromptNode(
            model_name_or_path="command", api_key=API_KEY, default_prompt_template=question_answering_with_references
        )
    elif provider == "huggingface":
        # TODO: swap out for meta-llama/Llama-2-7b-chat-hf or the 40b model once supported in Haystack+HF API free tier
        # The tiiuae/falcon-7b-instruct model cannot handle a complex prompt with references, so we use a very simple one
        simple_QA = PromptTemplate(
            prompt="deepset/question-answering", output_parser=AnswerParser(reference_pattern=r"Document\[(\d+)\]")
        )
        prompt_node = PromptNode(
            model_name_or_path="tiiuae/falcon-7b-instruct", api_key=API_KEY, default_prompt_template=simple_QA
        )
    elif provider == "openai":
        prompt_node = PromptNode(
            model_name_or_path="gpt-3.5-turbo-0301",
            api_key=API_KEY,
            default_prompt_template=question_answering_with_references,
        )
    else:
        logger.error('Given <provider> unknown. Please use any of "anthropic", "cohere", "huggingface", or "openai"')
    # Compose the query pipeline
    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
    query_pipeline.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

    return query_pipeline


def add_example_data(document_store, dir):
    # Importing top-level causes a circular import
    from haystack.nodes import TextConverter, PreProcessor

    if dir == "data/GoT_getting_started":
        # Download and add Game of Thrones TXT files
        fetch_archive_from_http(
            url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip",
            output_dir=dir,
        )
        files_to_index = [dir + "/" + f for f in os.listdir(dir)]
        converter = TextConverter(remove_numeric_tables=True, valid_languages=["en"])
        docs = [converter.convert(file_path=file, meta=None)[0] for file in files_to_index]
    else:
        # Here you can add a local folder with your files(.txt, .pdf, .docx).
        # You might need to install additional packages with "pip install farm-haystack[ocr,preprocessing,file-conversion,pdf]".
        # For more details, see: https://haystack.deepset.ai/tutorials/08_preprocessing.
        # Be aware that some of your data will be sent to external APIs if you use this functionality!
        files_to_index = [dir + "/" + f for f in os.listdir(dir)]
        logger.info("Adding %s number of files from local disk at %s.", len(files_to_index), dir)
        docs = convert_files_to_docs(dir_path=dir)

    preprocessor = PreProcessor(
        split_by="word", split_length=200, split_overlap=0, split_respect_sentence_boundary=True
    )
    docs_processed = preprocessor.process(docs)

    document_store.write_documents(documents=docs_processed)
