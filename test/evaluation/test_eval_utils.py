from haystack.dataclasses import Document, GeneratedAnswer
from haystack import Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.evaluation.eval_utils import (
    convert_component_output_from_dict,
    convert_component_output_to_dict,
    convert_component_outputs_from_dict,
    convert_component_outputs_to_dict,
    convert_pipeline_outputs_from_dict,
    convert_pipeline_outputs_to_dict,
)


def test_convert_list_documents_output():
    """
    Test serialization of outputs to dictionary and deserialization to dataclass  for a component that returns a list of Documents.
    """
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)
    retriever = InMemoryBM25Retriever(document_store=document_store)
    output = retriever.run("Who lives in Paris?")
    dict_serialized_output = convert_component_output_to_dict(output)
    deserialized_output = convert_component_output_from_dict(dict_serialized_output)

    assert output == deserialized_output


def test_convert_list_strings_output():
    """
    Test serialization of outputs to dictionary and deserialization to dataclass  for a component that returns a list of strings.
    """
    generator_output = {"replies": ["Jean lives in Paris."]}
    dict_serialized_output = convert_component_output_to_dict(generator_output)
    deserialized_output = convert_component_output_from_dict(dict_serialized_output)

    assert generator_output == deserialized_output


def test_convert_list_generated_answers_outputs():
    """
    Test serialization of outputs to dictionary and deserialization to dataclass  for a component that returns a list of Generated Answers.
    """
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]

    answer_builder = AnswerBuilder()
    output = answer_builder.run(query="Who lives in Paris?", replies=["Jean"], metadata=[{}], documents=documents)
    dict_serialized_output = convert_component_output_to_dict(output)
    deserialized_output = convert_component_output_from_dict(dict_serialized_output)

    assert output == deserialized_output


def test_convert_component_list_documents_outputs():
    """
    Test serialization of multiple component outputs to dictionary and deserialization to dataclass for a component that returns a list of Documents.
    """
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)
    retriever = InMemoryBM25Retriever(document_store=document_store)
    inputs = ["Who lives in Paris?", "Who lives in Berlin?", "Who lives in Rome?"]
    outputs = []
    for input in inputs:
        outputs.append(retriever.run(input))
    dict_serialized_outputs = convert_component_outputs_to_dict(outputs)
    deserialized_outputs = convert_component_outputs_from_dict(dict_serialized_outputs)

    assert outputs == deserialized_outputs


def test_convert_component_list_strings_outputs():
    """
    Test serialization of multiple component outputs to dictionary and deserialization to dataclass for a component that returns a list of strings.
    """
    generator_outputs = [{"replies": ["Jean"]}, {"replies": ["Mark"]}, {"replies": ["Giorgio"]}]
    dict_serialized_outputs = convert_component_outputs_to_dict(generator_outputs)
    deserialized_outputs = convert_component_outputs_from_dict(dict_serialized_outputs)

    assert generator_outputs == deserialized_outputs


def test_convert_component_list_generated_answer_outputs():
    """
    Test serialization of multiple component outputs to dictionary and deserialization to dataclass for a component that returns a list of Generated Answers.
    """
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]
    answer_builder = AnswerBuilder()
    questions = ["Who lives in Paris?", "Who lives in Berlin?", "Who lives in Rome?"]
    replies = [["Jean"], ["Mark"], ["Giorgio"]]
    outputs = []
    for question, reply in zip(questions, replies):
        outputs.append(answer_builder.run(query=question, replies=reply, metadata=[{}], documents=documents))
    dict_serialized_outputs = convert_component_outputs_to_dict(outputs)
    deserialized_outputs = convert_component_outputs_from_dict(dict_serialized_outputs)

    assert outputs == deserialized_outputs


def test_convert_pipeline_outputs_list_docs_to_dict():
    """
    Test serialization of multiple pipeline outputs to dictionary and deserialization to dataclass for a pipeline that returns a list of Documents.
    """
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]

    retrieval_pipeline = Pipeline()
    retrieval_pipeline.add_component(
        instance=InMemoryBM25Retriever(document_store=InMemoryDocumentStore()), name="retriever"
    )
    document_store = retrieval_pipeline.get_component("retriever").document_store
    document_store.write_documents(documents)
    questions = ["Who lives in Paris?", "Who lives in Berlin?", "Who lives in Rome?"]
    inputs = [{"retriever": {"query": question}} for question in questions]

    outputs = []
    for input in inputs:
        outputs.append(retrieval_pipeline.run(input))

    dict_serialized_outputs = convert_pipeline_outputs_to_dict(outputs)
    deserialized_outputs = convert_pipeline_outputs_from_dict(dict_serialized_outputs)

    assert outputs == deserialized_outputs


def test_convert_pipeline_outputs_list_str_to_dict():
    """
    Test serialization of multiple pipeline outputs to dictionary and deserialization to dataclass for a pipeline that returns a list of strings.
    """
    pipeline_outputs = [
        {"llm": {"replies": ["Jean"]}},
        {"llm": {"replies": ["Mark"]}},
        {"llm": {"replies": ["Giorgio"]}},
    ]
    dict_serialized_outputs = convert_pipeline_outputs_to_dict(pipeline_outputs)
    deserialized_outputs = convert_pipeline_outputs_from_dict(dict_serialized_outputs)

    assert pipeline_outputs == deserialized_outputs


def test_convert_pipeline_outputs_list_generated_answer_to_dict():
    """
    Test serialization of multiple pipeline outputs to dictionary and deserialization to dataclass for a pipeline that returns a list of Generated Answers.
    """
    pipeline_outputs = [
        {
            "answer_builder": {
                "answers": [GeneratedAnswer(data="Jean", query="Who lives in Paris?", documents=[], meta={})]
            }
        },
        {
            "answer_builder": {
                "answers": [GeneratedAnswer(data="Mark", query="Who lives in Berlin?", documents=[], meta={})]
            }
        },
        {
            "answer_builder": {
                "answers": [GeneratedAnswer(data="Giorgio", query="Who lives in Rome?", documents=[], meta={})]
            }
        },
    ]

    dict_serialized_outputs = convert_pipeline_outputs_to_dict(pipeline_outputs)
    deserialized_outputs = convert_pipeline_outputs_from_dict(dict_serialized_outputs)

    assert pipeline_outputs == deserialized_outputs
