import os.path

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from haystack.schema import Document, Label, MultiLabel
from haystack.utils import launch_es
from haystack.nodes import (
    EmbeddingRetriever,
    ElasticsearchRetriever,
    JoinDocuments,
    QuestionGenerator,
    Doc2QueryExpander,
)
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.pipelines import Pipeline

DOC_INDEX = "docs"


def init_doc_store(launch=False, embedding_dim=768, similarity="cosine", recreate_index=True, retriever_name=None):
    if launch:
        launch_es()

    document_store = ElasticsearchDocumentStore(
        host="localhost",
        username="",
        password="",
        index=DOC_INDEX,
        embedding_field="emb",
        embedding_dim=embedding_dim,
        excluded_meta_data=["emb"],
        analyzer="english",
        similarity=similarity,
        recreate_index=recreate_index,
        search_fields=["content", "questions"] if retriever_name == "doc2query" else "content",
    )

    return document_store


def qrels_to_haystack(corpus, queries, qrels):
    documents = {
        key: Document(content=val["text"], id=key, meta={"title": val.get("title", "")}) for key, val in corpus.items()
    }
    labels = []

    for query_id, query in queries.items():
        qrel = qrels[query_id]
        query_labels = []
        for doc_id in qrel.keys():
            document = documents.get(doc_id, None)
            if document is not None:
                label = Label(
                    query=query,
                    document=document,
                    is_correct_document=True,
                    is_correct_answer=False,
                    answer=None,
                    origin="gold-label",
                    id=query_id,
                )
            else:
                print(f"{doc_id} is missing")
            query_labels.append(label)
        labels.append(MultiLabel(query_labels, id))

    return list(documents.values()), labels


def load_beir_dataset(dataset_name, save_path):
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
    data_path = util.download_and_unzip(url, save_path)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    return corpus, queries, qrels


def index_documents(documents, document_store, retriever_name, max_seq_len, recreate_index=True):
    if recreate_index and retriever_name != "doc2query":
        document_store.write_documents(documents=documents, index=DOC_INDEX)

    if retriever_name == "doc2query":
        pipeline = Pipeline()
        question_generator = QuestionGenerator()
        doc2query = Doc2QueryExpander(target_field="questions")

        pipeline.add_node(question_generator, "QuestionGenerator", inputs=["File"])
        pipeline.add_node(doc2query, "Doc2Query", inputs=["QuestionGenerator"])
        pipeline.add_node(document_store, "ES", inputs=["Doc2Query"])
        print("Starting to run doc2query indexing")
        pipeline.run(documents=documents)
        retriever = ElasticsearchRetriever(document_store=document_store)
    elif retriever_name == "bm25":
        retriever = ElasticsearchRetriever(document_store=document_store)
    else:
        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model=retriever_name,
            max_seq_len=max_seq_len,
            model_format="sentence_transformers",
            batch_size=1 if retriever_name in ["gtr-t5-xxl", "sentence-t5-xxl"] else 16,
        )
        if recreate_index:
            document_store.update_embeddings(retriever=retriever, index=DOC_INDEX)

    return retriever


def evaluate_ensembled_pipeline(
    dataset_name, dataset_path, retriever_name, top_k, max_seq_len, emb_dim, es_launch=False, recreate_index=True
):
    document_store = init_doc_store(launch=es_launch, embedding_dim=emb_dim, recreate_index=recreate_index)
    corpus, queries, qrels = load_beir_dataset(dataset_name, save_path=dataset_path)
    documents, labels = qrels_to_haystack(corpus, queries, qrels)
    retriever = index_documents(
        documents, document_store, retriever_name=retriever_name, max_seq_len=max_seq_len, recreate_index=recreate_index
    )
    retriever_bm25 = index_documents(
        documents, document_store, retriever_name="bm25", max_seq_len=max_seq_len, recreate_index=False
    )
    joiner = JoinDocuments(join_mode="reciprocal_rank_fusion")

    pipeline = Pipeline()
    pipeline.add_node(retriever, "EmbeddingRetriever", ["Query"])
    pipeline.add_node(retriever_bm25, "ElasticRetriever", inputs=["Query"])
    pipeline.add_node(joiner, "Joiner", inputs=["EmbeddingRetriever", "ElasticRetriever"])

    result = pipeline.eval(
        labels=labels,
        params={
            "EmbeddingRetriever": {"top_k": top_k},
            "ElasticRetriever": {"top_k": top_k},
            "Joiner": {"top_k_join": top_k},
        },
    )
    metrics = result.calculate_metrics()
    print(metrics)
    return metrics


if __name__ == "__main__":
    dataset_name = "scifact"
    dataset_path = os.path.join(os.getcwd(), "../data")
    top_k = 10
    model_name = "all-mpnet-base-v2"
    es_launch = True
    max_seq_len = 384
    emb_dim = 768
    recreate_index = True

    params = {"dataset": dataset_name, "top_k": top_k, "model_name": model_name, "max_seq_len": max_seq_len}

    metrics = evaluate_ensembled_pipeline(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        retriever_name=model_name,
        top_k=top_k,
        max_seq_len=max_seq_len,
        es_launch=es_launch,
        emb_dim=emb_dim,
        recreate_index=recreate_index,
    )

    with open("report.md", "w") as text_file:
        text_file.write(f"# BEIR Evaluation\n Params:\n {params} \nMetrics: \n {metrics}")
