from typing import List
import requests
import pandas as pd
from haystack import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import RAGenerator, DensePassageRetriever
from haystack.utils import print_answers


def tutorial7_rag_generator():
    # Add documents from which you want generate answers
    # Download a csv containing some sample documents data
    # Here some sample documents data
    temp = requests.get(
        "https://raw.githubusercontent.com/deepset-ai/haystack/master/tutorials/small_generator_dataset.csv"
    )
    open("small_generator_dataset.csv", "wb").write(temp.content)

    # Get dataframe with columns "title", and "text"
    df = pd.read_csv("small_generator_dataset.csv", sep=",")
    # Minimal cleaning
    df.fillna(value="", inplace=True)

    print(df.head())

    titles = list(df["title"].values)
    texts = list(df["text"].values)

    # Create to haystack document format
    documents: List[Document] = []
    for title, text in zip(titles, texts):
        documents.append(Document(content=text, meta={"name": title or ""}))

    # Initialize FAISS document store to documents and corresponding index for embeddings
    # Set `return_embedding` to `True`, so generator doesn't have to perform re-embedding
    # Don't forget to install FAISS dependencies with `pip install farm-haystack[faiss]`
    document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True)

    # Initialize DPR Retriever to encode documents, encode question and query documents
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=True,
        embed_title=True,
    )

    # Initialize RAG Generator
    generator = RAGenerator(
        model_name_or_path="facebook/rag-token-nq",
        use_gpu=True,
        top_k=1,
        max_length=200,
        min_length=2,
        embed_title=True,
        num_beams=2,
    )

    # Delete existing documents in documents store
    document_store.delete_documents()
    # Write documents to document store
    document_store.write_documents(documents)
    # Add documents embeddings to index
    document_store.update_embeddings(retriever=retriever)

    # Now ask your questions
    # We have some sample questions
    QUESTIONS = [
        "who got the first nobel prize in physics",
        "when is the next deadpool movie being released",
        "which mode is used for short wave broadcast service",
        "who is the owner of reading football club",
        "when is the next scandal episode coming out",
        "when is the last time the philadelphia won the superbowl",
        "what is the most current adobe flash player version",
        "how many episodes are there in dragon ball z",
        "what is the first step in the evolution of the eye",
        "where is gall bladder situated in human body",
        "what is the main mineral in lithium batteries",
        "who is the president of usa right now",
        "where do the greasers live in the outsiders",
        "panda is a national animal of which country",
        "what is the name of manchester united stadium",
    ]

    # Now generate answer for question
    for question in QUESTIONS:
        # Retrieve related documents from retriever
        retriever_results = retriever.retrieve(query=question)

        # Now generate answer from question and retrieved documents
        predicted_result = generator.predict(query=question, documents=retriever_results, top_k=1)

        # Print you answer
        answers = predicted_result["answers"]
        print(f" -> Generated answer is '{answers[0].answer}' for the question = '{question}'")

    # Or alternatively use the Pipeline class
    from haystack.pipelines import GenerativeQAPipeline

    pipe = GenerativeQAPipeline(generator=generator, retriever=retriever)
    for question in QUESTIONS:
        res = pipe.run(query=question, params={"Generator": {"top_k": 1}, "Retriever": {"top_k": 5}})
        print_answers(res, details="minimum")


if __name__ == "__main__":
    tutorial7_rag_generator()

# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/
