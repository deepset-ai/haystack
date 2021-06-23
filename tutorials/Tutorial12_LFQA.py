from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.generator.transformers import Seq2SeqGenerator


def tutorial12_lfqa():

    """
    Document Store:
    FAISS is a library for efficient similarity search on a cluster of dense vectors.
    The `FAISSDocumentStore` uses a SQL(SQLite in-memory be default) database under-the-hood
    to store the document text and other meta data. The vector embeddings of the text are
    indexed on a FAISS Index that later is queried for searching answers.
    The default flavour of FAISSDocumentStore is "Flat" but can also be set to "HNSW" for
    faster search at the expense of some accuracy. Just set the faiss_index_factor_str argument in the constructor.
    For more info on which suits your use case: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    """

    from haystack.document_store.faiss import FAISSDocumentStore

    document_store = FAISSDocumentStore(vector_dim=128, faiss_index_factory_str="Flat")

    """
    Cleaning & indexing documents:
    Similarly to the previous tutorials, we download, convert and index some Game of Thrones articles to our DocumentStore
    """

    # Let's first get some files that we want to use
    doc_dir = "data/article_txt_got"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # Convert files to dicts
    dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

    # Now, let's write the dicts containing documents to our DB.
    document_store.write_documents(dicts)

    """
    Initalize Retriever and Reader/Generator:
    We use a `RetribertRetriever` and we invoke `update_embeddings` to index the embeddings of documents in the `FAISSDocumentStore`
    """

    from haystack.retriever.dense import EmbeddingRetriever

    retriever = EmbeddingRetriever(document_store=document_store,
                                   embedding_model="yjernite/retribert-base-uncased",
                                   model_format="retribert")

    document_store.update_embeddings(retriever)

    """Before we blindly use the `RetribertRetriever` let's empirically test it to make sure a simple search indeed finds the relevant documents."""

    from haystack.utils import print_answers, print_documents
    from haystack.pipeline import DocumentSearchPipeline

    p_retrieval = DocumentSearchPipeline(retriever)
    res = p_retrieval.run(
        query="Tell me something about Arya Stark?",
        top_k_retriever=5
    )
    print_documents(res, max_text_len=512)

    """
    Similar to previous Tutorials we now initalize our reader/generator.
    Here we use a `Seq2SeqGenerator` with the *yjernite/bart_eli5* model (see: https://huggingface.co/yjernite/bart_eli5)
    """

    generator = Seq2SeqGenerator(model_name_or_path="yjernite/bart_eli5")

    """
    Pipeline:
    With a Haystack `Pipeline` you can stick together your building blocks to a search pipeline.
    Under the hood, `Pipelines` are Directed Acyclic Graphs (DAGs) that you can easily customize for your own use cases.
    To speed things up, Haystack also comes with a few predefined Pipelines. One of them is the `GenerativeQAPipeline` that combines a retriever and a reader/generator to answer our questions.
    You can learn more about `Pipelines` in the [docs](https://haystack.deepset.ai/docs/latest/pipelinesmd).
    """

    from haystack.pipeline import GenerativeQAPipeline
    pipe = GenerativeQAPipeline(generator, retriever)

    """Voilà! Ask a question!"""

    query_1 = "Why did Arya Stark's character get portrayed in a television adaptation?"
    result_1 = pipe.run(query=query_1, top_k_retriever=1)
    print(f"Query: {query_1}")
    print(f"Answer: {result_1['answers'][0]}")
    print()

    query_2 = "What kind of character does Arya Stark play?"
    result_2 = pipe.run(query=query_2, top_k_retriever=1)
    print(f"Query: {query_2}")
    print(f"Answer: {result_2['answers'][0]}")
    print()
    pipe.run(query=query_2, top_k_retriever=1)


if __name__ == "__main__":
    tutorial12_lfqa()


# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/