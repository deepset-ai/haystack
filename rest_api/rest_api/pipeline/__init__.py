from typing import Any, Dict

import os
import torch
import logging
from pathlib import Path

from haystack.pipelines.base import Pipeline
from haystack.document_stores import FAISSDocumentStore, InMemoryDocumentStore
from haystack.errors import PipelineConfigError

from rest_api.controller.utils import RequestLimiter
from rest_api.schema import PipelineHyperParams

from haystack import BaseComponent
from haystack.nodes import BM25Retriever, FARMReader, EmbeddingRetriever, JoinAnswers, Docs2Answers
from typing import List
from haystack.document_stores import ElasticsearchDocumentStore


logger = logging.getLogger(__name__)

# Since each instance of FAISSDocumentStore creates an in-memory FAISS index, the Indexing & Query Pipelines would
# end up with different indices. The same applies for InMemoryDocumentStore.
UNSUPPORTED_DOC_STORES = (FAISSDocumentStore, InMemoryDocumentStore)


def setup_pipelines(pipeline_hyper_params: PipelineHyperParams) -> Dict[str, Any]:
    # Re-import the configuration variables
    from rest_api import config  # pylint: disable=reimported

    pipelines = {}

    # ------------------
    document_store = ElasticsearchDocumentStore()
    extractive_document_store = ElasticsearchDocumentStore(
        index="rulebook",
        embedding_dim=pipeline_hyper_params.extractive_embedding_dim,
        similarity=pipeline_hyper_params.extractive_similarity_function,
    )
    faq_document_store = ElasticsearchDocumentStore(
        index="faq",
        embedding_dim=pipeline_hyper_params.faq_embedding_dim,
        similarity=pipeline_hyper_params.faq_similarity_function,
    )

    extractive_reader_option = pipeline_hyper_params.extractive_reader_option
    faq_retriever_option = pipeline_hyper_params.faq_retriever_option

    faq_retriever = EmbeddingRetriever(
        document_store=faq_document_store,
        embedding_model=faq_retriever_option,
        use_gpu=torch.cuda.is_available(),
        scale_score=False,
        top_k=pipeline_hyper_params.top_k,
    )

    faq_document_store.update_embeddings(faq_retriever, index="faq")

    ext_retriever = BM25Retriever(document_store=extractive_document_store, top_k=pipeline_hyper_params.top_k)
    ext_reader = FARMReader(
        model_name_or_path=extractive_reader_option,
        use_gpu=torch.cuda.is_available(),
        top_k=pipeline_hyper_params.top_k,
    )

    class CustomQueryClassifier(BaseComponent):
        outgoing_edges = 2

        def run(self, query: str, index: str):
            if index == "faq":
                return {}, "output_1"
            else:
                return {}, "output_2"

        def run_batch(self, queries: List[str], index: str):
            split = {"output_1": {"queries": []}, "output_2": {"queries": []}}
            for query in queries:
                if index == "faq":
                    split["output_1"]["queries"].append(query)
                else:
                    split["output_2"]["queries"].append(query)

            return split, "split"

    query_pipeline = Pipeline()
    query_pipeline.add_node(component=CustomQueryClassifier(), name="CustomClassifier", inputs=["Query"])
    query_pipeline.add_node(component=faq_retriever, name="FaqRetriever", inputs=["CustomClassifier.output_1"])
    query_pipeline.add_node(component=Docs2Answers(), name="Docs2Answers", inputs=["FaqRetriever"])
    query_pipeline.add_node(component=ext_retriever, name="ExtrRetriever", inputs=["CustomClassifier.output_2"])
    query_pipeline.add_node(component=ext_reader, name="ExtrReader", inputs=["ExtrRetriever"])
    query_pipeline.add_node(
        component=JoinAnswers(join_mode="concatenate", sort_by_score=False),
        name="JoinResults",
        inputs=["ExtrReader", "Docs2Answers"],
    )
    # ------------------

    logging.info(f"Loaded pipeline nodes: {query_pipeline.graph.nodes.keys()}")
    pipelines["query_pipeline"] = query_pipeline

    # Find document store
    # document_store = query_pipeline.get_document_store()
    logging.info(f"Loaded docstore: {document_store}")
    pipelines["document_store"] = document_store

    # Setup concurrency limiter
    concurrency_limiter = RequestLimiter(config.CONCURRENT_REQUEST_PER_WORKER)
    logging.info("Concurrent requests per worker: {CONCURRENT_REQUEST_PER_WORKER}")
    pipelines["concurrency_limiter"] = concurrency_limiter

    # Load indexing pipeline (if available)
    try:
        indexing_pipeline = Pipeline.load_from_yaml(
            Path(config.PIPELINE_YAML_PATH), pipeline_name=config.INDEXING_PIPELINE_NAME
        )
        docstore = indexing_pipeline.get_document_store()
        if isinstance(docstore, UNSUPPORTED_DOC_STORES):
            indexing_pipeline = None
            raise PipelineConfigError(
                "Indexing pipelines with FAISSDocumentStore or InMemoryDocumentStore are not supported by the REST APIs."
            )

    except PipelineConfigError as e:
        indexing_pipeline = None
        logger.error(f"{e.message}\nFile Upload API will not be available.")

    finally:
        pipelines["indexing_pipeline"] = indexing_pipeline

    # Create directory for uploaded files
    os.makedirs(config.FILE_UPLOAD_PATH, exist_ok=True)

    return pipelines
