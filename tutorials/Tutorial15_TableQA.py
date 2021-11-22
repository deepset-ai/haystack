import json

import pandas as pd

from haystack.utils import launch_es, fetch_archive_from_http, print_answers
from haystack.document_stores import ElasticsearchDocumentStore
from haystack import Document, Pipeline
from haystack.nodes.retriever import TableTextRetriever
from haystack.nodes import TableReader


def tutorial15_tableqa():

    # Recommended: Start Elasticsearch using Docker via the Haystack utility function
    launch_es()

    ## Connect to Elasticsearch
    # We want to use a small model producing 512-dimensional embeddings, so we need to set embedding_dim to 512
    document_store = ElasticsearchDocumentStore(host="localhost",
                                                username="",
                                                password="",
                                                index="document",
                                                embedding_dim=512)

    ## Add Tables to DocumentStore

    # Let's first fetch some tables that we want to query
    # Here: 1000 tables from OTT-QA

    doc_dir = "data"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/ottqa_tables_sample.json.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # Add the tables to the DocumentStore
    def read_ottqa_tables(filename):
        processed_tables = []
        with open(filename) as tables:
            tables = json.load(tables)
            for key, table in tables.items():
                current_columns = table["header"]
                current_rows = table["data"]
                current_df = pd.DataFrame(columns=current_columns, data=current_rows)
                current_doc_title = table["title"]
                current_section_title = table["section_title"]
                document = Document(
                    content=current_df,
                    content_type="table",
                    meta={"title": current_doc_title, "section_title": current_section_title},
                    id=key
                )
                processed_tables.append(document)

        return processed_tables


    tables = read_ottqa_tables("data/ottqa_tables_sample.json")
    document_store.write_documents(tables, index="document")


    ### Retriever

    # Retrievers help narrowing down the scope for the Reader to a subset of tables where a given question could be answered.
    # They use some simple but fast algorithm.
    #
    # **Here:** We use the TableTextRetriever capable of retrieving relevant content among a database
    # of texts and tables using dense embeddings.

    retriever = TableTextRetriever(
        document_store=document_store,
        query_embedding_model="deepset/bert-small-mm_retrieval-question_encoder",
        passage_embedding_model="deepset/bert-small-mm_retrieval-passage_encoder",
        table_embedding_model="deepset/bert-small-mm_retrieval-table_encoder",
        embed_meta_fields=["title", "section_title"]
    )

    # Add table embeddings to the tables in DocumentStore
    document_store.update_embeddings(retriever=retriever)

    ## Alternative: ElasticsearchRetriever
    #from haystack.nodes.retriever import ElasticsearchRetriever
    #retriever = ElasticsearchRetriever(document_store=document_store)

    # Try the Retriever
    from haystack.utils import print_documents

    retrieved_tables = retriever.retrieve("How many twin buildings are under construction?", top_k=5)
    # Get highest scored table
    print(retrieved_tables[0].content)

    ### Reader
    # The TableReader is based on TaPas, a transformer-based language model capable of grasping the two-dimensional structure of a table.
    # It scans the tables returned by the retriever and extracts the anser.
    # The available TableReader models can be found [here](https://huggingface.co/models?pipeline_tag=table-question-answering&sort=downloads).
    #
    # **Notice**: The TableReader will return an answer for each table, even if the query cannot be answered by the table.
    # Furthermore, the confidence scores are not useful as of now, given that they will *always* be very high (i.e. 1 or close to 1).


    reader = TableReader(model_name_or_path="google/tapas-base-finetuned-wtq", max_seq_len=512)

    # Try the TableReader on one Table (highest-scored retrieved table)

    table_doc = document_store.get_document_by_id("List_of_tallest_twin_buildings_and_structures_in_the_world_1")
    print(table_doc.content)

    prediction = reader.predict(query="How many twin buildings are under construction?", documents=[table_doc])
    print_answers(prediction, details="minimum")

    ### Pipeline
    # The Retriever and the Reader can be sticked together to a pipeline in order to first retrieve relevant tables
    # and then extract the answer.
    #
    # **Notice**: Given that the `TableReader` does not provide useful confidence scores and returns an answer
    # for each of the tables, the sorting of the answers might be not helpful.


    table_qa_pipeline = Pipeline()
    table_qa_pipeline.add_node(component=retriever, name="TableTextRetriever", inputs=["Query"])
    table_qa_pipeline.add_node(component=reader, name="TableReader", inputs=["TableTextRetriever"])

    prediction = table_qa_pipeline.run("How many twin buildings are under construction?")
    print_answers(prediction, details="minimum")


if __name__ == "__main__":
    tutorial15_tableqa()

# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/
