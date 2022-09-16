import logging

# We configure how logging messages should be displayed and which log level should be used before importing Haystack.
# Example log message:
# INFO - haystack.utils.preprocessing -  Converting data/tutorial1/218_Olenna_Tyrell.txt
# Default log level in basicConfig is WARNING so the explicit parameter is not necessary but can be changed easily:
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

import os
import json
import time

import pandas as pd

from haystack import Label, MultiLabel, Answer
from haystack.utils import launch_es, fetch_archive_from_http, print_answers
from haystack.document_stores import ElasticsearchDocumentStore
from haystack import Document, Pipeline
from haystack.nodes.retriever import EmbeddingRetriever
from haystack.nodes import TableReader, FARMReader, RouteDocuments, JoinAnswers, ParsrConverter


def tutorial15_tableqa():

    # Recommended: Start Elasticsearch using Docker via the Haystack utility function
    launch_es()

    ## Connect to Elasticsearch
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

    ## Add Tables to DocumentStore

    # Let's first fetch some tables that we want to query
    # Here: 1000 tables + texts

    doc_dir = "data/tutorial15"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/table_text_dataset.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # Add the tables to the DocumentStore
    def read_tables(filename):
        processed_tables = []
        with open(filename) as tables:
            tables = json.load(tables)
            for key, table in tables.items():
                current_columns = table["header"]
                current_rows = table["data"]
                current_df = pd.DataFrame(columns=current_columns, data=current_rows)
                document = Document(content=current_df, content_type="table", id=key)
                processed_tables.append(document)

        return processed_tables

    tables = read_tables(f"{doc_dir}/tables.json")
    document_store.write_documents(tables, index="document")

    ### Retriever

    # Retrievers help narrowing down the scope for the Reader to a subset of tables where a given question could be answered.
    # They use some simple but fast algorithm.
    #
    # **Here:** We use the EmbeddingRetriever capable of retrieving relevant content among a database
    # of texts and tables using dense embeddings.

    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/all-mpnet-base-v2-table")

    # Add table embeddings to the tables in DocumentStore
    document_store.update_embeddings(retriever=retriever)

    ## Alternative: BM25Retriever
    # from haystack.nodes.retriever import BM25Retriever
    # retriever = BM25Retriever(document_store=document_store)

    # Try the Retriever
    from haystack.utils import print_documents

    retrieved_tables = retriever.retrieve("Who won the Super Bowl?", top_k=5)
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

    # Try the TableReader on one Table

    table_doc = document_store.get_document_by_id("36964e90-3735-4ba1-8e6a-bec236e88bb2")
    print(table_doc.content)

    prediction = reader.predict(query="Who played Gregory House in the series House?", documents=[table_doc])
    print_answers(prediction, details="minimum")

    ### Pipeline
    # The Retriever and the Reader can be sticked together to a pipeline in order to first retrieve relevant tables
    # and then extract the answer.
    #
    # **Notice**: Given that the `TableReader` does not provide useful confidence scores and returns an answer
    # for each of the tables, the sorting of the answers might be not helpful.

    table_qa_pipeline = Pipeline()
    table_qa_pipeline.add_node(component=retriever, name="EmbeddingRetriever", inputs=["Query"])
    table_qa_pipeline.add_node(component=reader, name="TableReader", inputs=["EmbeddingRetriever"])

    prediction = table_qa_pipeline.run("When was Guilty Gear Xrd : Sign released?")
    print_answers(prediction, details="minimum")

    ### Pipeline for QA on Combination of Text and Tables
    # We are using one node for retrieving both texts and tables, the EmbeddingRetriever.
    # In order to do question-answering on the Documents coming from the EmbeddingRetriever, we need to route
    # Documents of type "text" to a FARMReader ( or alternatively TransformersReader) and Documents of type
    # "table" to a TableReader.

    text_reader = FARMReader("deepset/roberta-base-squad2")
    # In order to get meaningful scores from the TableReader, use "deepset/tapas-large-nq-hn-reader" or
    # "deepset/tapas-large-nq-reader" as TableReader models. The disadvantage of these models is, however,
    # that they are not capable of doing aggregations over multiple table cells.
    table_reader = TableReader("deepset/tapas-large-nq-hn-reader")
    route_documents = RouteDocuments()
    join_answers = JoinAnswers()

    text_table_qa_pipeline = Pipeline()
    text_table_qa_pipeline.add_node(component=retriever, name="EmbeddingRetriever", inputs=["Query"])
    text_table_qa_pipeline.add_node(component=route_documents, name="RouteDocuments", inputs=["EmbeddingRetriever"])
    text_table_qa_pipeline.add_node(component=text_reader, name="TextReader", inputs=["RouteDocuments.output_1"])
    text_table_qa_pipeline.add_node(component=table_reader, name="TableReader", inputs=["RouteDocuments.output_2"])
    text_table_qa_pipeline.add_node(component=join_answers, name="JoinAnswers", inputs=["TextReader", "TableReader"])

    # Add texts to the document store
    def read_texts(filename):
        processed_passages = []
        with open(filename) as passages:
            passages = json.load(passages)
            for key, content in passages.items():
                document = Document(content=content, content_type="text", id=key)
                processed_passages.append(document)

        return processed_passages

    passages = read_texts(f"{doc_dir}/texts.json")
    document_store.write_documents(passages)

    document_store.update_embeddings(retriever=retriever, update_existing_embeddings=False)

    # Example query whose answer resides in a text passage
    predictions = text_table_qa_pipeline.run(query="Which country does the film Macaroni come from?")
    # We can see both text passages and tables as contexts of the predicted answers.
    print_answers(predictions, details="minimum")

    # Example query whose answer resides in a table
    predictions = text_table_qa_pipeline.run(query="Who was Thomas Alva Edison?")
    # We can see both text passages and tables as contexts of the predicted answers.
    print_answers(predictions, details="minimum")

    ### Evaluation
    # To evaluate our pipeline, we can use haystack's evaluation feature. We just need to convert our labels into `MultiLabel` objects and the `eval` method will do the rest.

    def read_labels(filename, tables):
        processed_labels = []
        with open(filename) as labels:
            labels = json.load(labels)
            for table in tables:
                if table.id not in labels:
                    continue
                label = labels[table.id]
                label = Label(
                    query=label["query"],
                    document=table,
                    is_correct_answer=True,
                    is_correct_document=True,
                    answer=Answer(answer=label["answer"]),
                    origin="gold-label",
                )
                processed_labels.append(MultiLabel(labels=[label]))
        return processed_labels

    table_labels = read_labels(f"{doc_dir}/labels.json", tables)
    passage_labels = read_labels(f"{doc_dir}/labels.json", passages)

    eval_results = text_table_qa_pipeline.eval(table_labels + passage_labels, params={"top_k": 10})

    # Calculating and printing the evaluation metrics
    print(eval_results.calculate_metrics())

    ## Adding tables from PDFs
    # It can sometimes be hard to provide your data in form of a pandas DataFrame.
    # For this case, we provide the `ParsrConverter` wrapper that can help you to convert, for example, a PDF file into a document that you can index.
    os.system("docker run -d -p 3001:3001 axarev/parsr")
    time.sleep(30)
    os.system("wget https://www.w3.org/WAI/WCAG21/working-examples/pdf-table/table.pdf")

    converter = ParsrConverter()
    docs = converter.convert("table.pdf")
    tables = [doc for doc in docs if doc.content_type == "table"]

    print(tables)


if __name__ == "__main__":
    tutorial15_tableqa()

# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/
