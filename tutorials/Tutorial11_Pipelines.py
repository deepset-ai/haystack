import logging

# We configure how logging messages should be displayed and which log level should be used before importing Haystack.
# Example log message:
# INFO - haystack.utils.preprocessing -  Converting data/tutorial1/218_Olenna_Tyrell.txt
# Default log level in basicConfig is WARNING so the explicit parameter is not necessary but can be changed easily:
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from haystack.utils import (
    clean_wiki_text,
    print_answers,
    print_documents,
    fetch_archive_from_http,
    convert_files_to_docs,
    launch_es,
)
from pprint import pprint
from haystack import Pipeline
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, EmbeddingRetriever, FARMReader, RAGenerator, BaseComponent, JoinDocuments
from haystack.pipelines import ExtractiveQAPipeline, DocumentSearchPipeline, GenerativeQAPipeline


def tutorial11_pipelines():
    # Download and prepare data - 517 Wikipedia articles for Game of Thrones
    doc_dir = "data/tutorial11"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt11.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # convert files to dicts containing documents that can be indexed to our datastore
    got_docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

    # Initialize DocumentStore and index documents
    launch_es()
    document_store = ElasticsearchDocumentStore()
    document_store.delete_documents()
    document_store.write_documents(got_docs)

    # Initialize Sparse retriever
    bm25_retriever = BM25Retriever(document_store=document_store)

    # Initialize dense retriever
    embedding_retriever = EmbeddingRetriever(
        document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
    )
    document_store.update_embeddings(embedding_retriever, update_existing_embeddings=False)

    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

    print()
    print("######################")
    print("# Prebuilt Pipelines #")
    print("######################")

    print()
    print("# Extractive QA Pipeline")
    print("########################")

    query = "Who is the father of Arya Stark?"
    p_extractive_premade = ExtractiveQAPipeline(reader=reader, retriever=bm25_retriever)
    res = p_extractive_premade.run(query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})
    print("\nQuery: ", query)
    print("Answers:")
    print_answers(res, details="minimum")

    print()
    print("# Document Search Pipeline")
    print("##########################")

    query = "Who is the father of Arya Stark?"
    p_retrieval = DocumentSearchPipeline(bm25_retriever)
    res = p_retrieval.run(query=query, params={"Retriever": {"top_k": 10}})
    print()
    print_documents(res, max_text_len=200)

    print()
    print("# Generator Pipeline")
    print("####################")

    # We set this to True so that the document store returns document embeddings
    # with each document, this is needed by the Generator
    document_store.return_embedding = True

    # Initialize generator
    rag_generator = RAGenerator()

    # Generative QA
    query = "Who is the father of Arya Stark?"
    p_generator = GenerativeQAPipeline(generator=rag_generator, retriever=embedding_retriever)
    res = p_generator.run(query=query, params={"Retriever": {"top_k": 10}})
    print()
    print_answers(res, details="minimum")

    # We are setting this to False so that in later pipelines,
    # we get a cleaner printout
    document_store.return_embedding = False

    ##############################
    # Creating Pipeline Diagrams #
    ##############################

    p_extractive_premade.draw("pipeline_extractive_premade.png")
    p_retrieval.draw("pipeline_retrieval.png")
    p_generator.draw("pipeline_generator.png")

    print()
    print("####################")
    print("# Custom Pipelines #")
    print("####################")

    print()
    print("# Extractive QA Pipeline")
    print("########################")

    # Custom built extractive QA pipeline
    p_extractive = Pipeline()
    p_extractive.add_node(component=bm25_retriever, name="Retriever", inputs=["Query"])
    p_extractive.add_node(component=reader, name="Reader", inputs=["Retriever"])

    # Now we can run it
    query = "Who is the father of Arya Stark?"
    res = p_extractive.run(query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})
    print("\nQuery: ", query)
    print("Answers:")
    print_answers(res, details="minimum")
    p_extractive.draw("pipeline_extractive.png")

    print()
    print("# Ensembled Retriever Pipeline")
    print("##############################")

    # Create ensembled pipeline
    p_ensemble = Pipeline()
    p_ensemble.add_node(component=bm25_retriever, name="ESRetriever", inputs=["Query"])
    p_ensemble.add_node(component=embedding_retriever, name="EmbeddingRetriever", inputs=["Query"])
    p_ensemble.add_node(
        component=JoinDocuments(join_mode="concatenate"),
        name="JoinResults",
        inputs=["ESRetriever", "EmbeddingRetriever"],
    )
    p_ensemble.add_node(component=reader, name="Reader", inputs=["JoinResults"])
    p_ensemble.draw("pipeline_ensemble.png")

    # Run pipeline
    query = "Who is the father of Arya Stark?"
    res = p_ensemble.run(
        query="Who is the father of Arya Stark?",
        params={"ESRetriever": {"top_k": 5}, "EmbeddingRetriever": {"top_k": 5}},
    )
    print("\nQuery: ", query)
    print("Answers:")
    print_answers(res, details="minimum")

    print()
    print("# Query Classification Pipeline")
    print("###############################")

    # Decision Nodes help you route your data so that only certain branches of your `Pipeline` are run.
    # Though this looks very similar to the ensembled pipeline shown above,
    # the key difference is that only one of the retrievers is run for each request.
    # By contrast both retrievers are always run in the ensembled approach.

    class CustomQueryClassifier(BaseComponent):
        outgoing_edges = 2

        def run(self, query):
            if "?" in query:
                return {}, "output_2"
            else:
                return {}, "output_1"

        def run_batch(self, queries):
            split = {"output_1": {"queries": []}, "output_2": {"queries": []}}
            for query in queries:
                if "?" in query:
                    split["output_2"]["queries"].append(query)
                else:
                    split["output_1"]["queries"].append(query)

            return split, "split"

    # Here we build the pipeline
    p_classifier = Pipeline()
    p_classifier.add_node(component=CustomQueryClassifier(), name="QueryClassifier", inputs=["Query"])
    p_classifier.add_node(component=bm25_retriever, name="ESRetriever", inputs=["QueryClassifier.output_1"])
    p_classifier.add_node(component=embedding_retriever, name="EmbeddingRetriever", inputs=["QueryClassifier.output_2"])
    p_classifier.add_node(component=reader, name="QAReader", inputs=["ESRetriever", "EmbeddingRetriever"])
    p_classifier.draw("pipeline_classifier.png")

    # Run only the dense retriever on the full sentence query
    query = "Who is the father of Arya Stark?"
    res_1 = p_classifier.run(query=query)
    print()
    print("\nQuery: ", query)
    print(" * Embedding Retriever Answers:")
    print_answers(res_1, details="minimum")

    # Run only the sparse retriever on a keyword based query
    query = "Arya Stark father"
    res_2 = p_classifier.run(query=query)
    print()
    print("\nQuery: ", query)
    print(" * ES Answers:")
    print_answers(res_2, details="minimum")

    print("#######################")
    print("# Debugging Pipelines #")
    print("#######################")
    # You can print out debug information from nodes in your pipelines in a few different ways.

    # 1) You can set the `debug` attribute of a given node.
    bm25_retriever.debug = True

    # 2) You can provide `debug` as a parameter when running your pipeline
    result = p_classifier.run(query="Who is the father of Arya Stark?", params={"ESRetriever": {"debug": True}})

    # 3) You can provide the `debug` parameter to all nodes in your pipeline
    result = p_classifier.run(query="Who is the father of Arya Stark?", params={"debug": True})

    pprint(result["_debug"])


if __name__ == "__main__":
    tutorial11_pipelines()

# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/
