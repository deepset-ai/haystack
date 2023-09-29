from haystack.preview import Pipeline, Document
from haystack.preview.document_stores import MemoryDocumentStore
from haystack.preview.components.retrievers import MemoryBM25Retriever
from haystack.preview.components.readers import ExtractiveReader


def test_extractive_qa_pipeline():
    document_store = MemoryDocumentStore()

    documents = [
        Document(text="My name is Jean and I live in Paris."),
        Document(text="My name is Mark and I live in Berlin."),
        Document(text="My name is Giorgio and I live in Rome."),
    ]

    document_store.write_documents(documents)

    qa_pipeline = Pipeline()
    qa_pipeline.add_component(instance=MemoryBM25Retriever(document_store=document_store), name="retriever")
    qa_pipeline.add_component(instance=ExtractiveReader(model_name_or_path="deepset/tinyroberta-squad2"), name="reader")
    qa_pipeline.connect("retriever", "reader")

    questions = ["Who lives in Paris?", "Who lives in Berlin?", "Who lives in Rome?"]
    answers_spywords = ["Jean", "Mark", "Giorgio"]

    for question, spyword, doc in zip(questions, answers_spywords, documents):
        result = qa_pipeline.run({"retriever": {"query": question}, "reader": {"query": question}})

        extracted_answers = result["reader"]["answers"]

        # we expect at least one real answer and no_answer
        assert len(extracted_answers) > 1

        # the best answer should contain the spyword
        assert spyword in extracted_answers[0].data

        # no_answer
        assert extracted_answers[-1].data is None

        # since these questions are easily answerable, the best answer should have higher probability than no_answer
        assert extracted_answers[0].probability >= extracted_answers[-1].probability

        for answer in extracted_answers:
            assert answer.query == question

            assert hasattr(answer, "probability")
            assert hasattr(answer, "start")
            assert hasattr(answer, "end")

            assert hasattr(answer, "document")
            # the answer is extracted from the correct document
            if answer.document is not None:
                assert answer.document == doc
