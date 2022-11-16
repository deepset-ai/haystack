import logging
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from haystack.document_stores import ElasticsearchDocumentStore

from haystack.nodes import EmbeddingRetriever
from haystack.utils import launch_es, print_answers, fetch_archive_from_http
import pandas as pd
from haystack.pipelines import FAQPipeline


def basic_faq_pipeline():

    launch_es()
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

    doc_dir = "data/basic_faq_pipeline"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/small_faq_covid.csv.zip"
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
    pipe = FAQPipeline(retriever=retriever)

    prediction = pipe.run(query="How is the virus spreading?", params={"Retriever": {"top_k": 10}})
    print_answers(prediction, details="medium")


if __name__ == "__main__":
    basic_faq_pipeline()