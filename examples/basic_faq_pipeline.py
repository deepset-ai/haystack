import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from haystack.document_stores import ElasticsearchDocumentStore

from haystack.nodes import EmbeddingRetriever
from haystack.nodes.other.docs2answers import Docs2Answers
from haystack.utils import launch_es, print_answers, fetch_archive_from_http
import pandas as pd
from haystack.pipelines import Pipeline


def basic_faq_pipeline():
    document_store = ElasticsearchDocumentStore(
        host="localhost",
        username="",
        password="",
        index="document",
        embedding_field="question_emb",
        embedding_dim=384,
        excluded_meta_data=["question_emb"],
        similarity="cosine",
    )

    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        use_gpu=True,
        scale_score=False,
    )

    doc_to_answers = Docs2Answers()

    doc_dir = "data/basic_faq_pipeline"
    s3_url = "https://core-engineering.s3.eu-central-1.amazonaws.com/public/scripts/small_faq_covid.csv1.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    df = pd.read_csv(f"{doc_dir}/small_faq_covid.csv")

    # Minimal cleaning
    df.fillna(value="", inplace=True)
    df["question"] = df["question"].apply(lambda x: x.strip())
    print(df.head())

    # Get embeddings for our questions from the FAQs
    questions = list(df["question"].values)
    df["question_emb"] = retriever.embed_queries(queries=questions).tolist()
    df = df.rename(columns={"question": "content"})

    # Convert Dataframe to list of dicts and index them in our DocumentStore
    docs_to_index = df.to_dict(orient="records")
    document_store.write_documents(docs_to_index)

    # Initialize a Pipeline (this time without a reader) and ask questions
    pipeline = Pipeline()
    pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    pipeline.add_node(component=doc_to_answers, name="Docs2Answers", inputs=["Retriever"])

    # Ask a question
    prediction = pipeline.run(query="How is the virus spreading?", params={"Retriever": {"top_k": 10}})

    print_answers(prediction, details="medium")
    return prediction


if __name__ == "__main__":
    launch_es()
    basic_faq_pipeline()
