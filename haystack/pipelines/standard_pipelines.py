import logging
from abc import ABC
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Dict
from functools import wraps

from haystack.schema import Document, EvaluationResult, MultiLabel
from haystack.nodes.answer_generator import BaseGenerator
from haystack.nodes.other import Docs2Answers
from haystack.nodes.reader import BaseReader
from haystack.nodes.retriever import BaseRetriever
from haystack.nodes.summarizer import BaseSummarizer
from haystack.nodes.translator import BaseTranslator
from haystack.nodes.question_generator import QuestionGenerator
from haystack.document_stores import BaseDocumentStore
from haystack.pipelines import Pipeline


logger = logging.getLogger(__name__)


class BaseStandardPipeline(ABC):
    """
    Base class for pre-made standard Haystack pipelines.
    This class does not inherit from Pipeline.
    """
    pipeline: Pipeline

    def add_node(self, component, name: str, inputs: List[str]):
        """
        Add a new node to the pipeline.

        :param component: The object to be called when the data is passed to the node. It can be a Haystack component
                          (like Retriever, Reader, or Generator) or a user-defined object that implements a run()
                          method to process incoming data from predecessor node.
        :param name: The name for the node. It must not contain any dots.
        :param inputs: A list of inputs to the node. If the predecessor node has a single outgoing edge, just the name
                       of node is sufficient. For instance, a 'ElasticsearchRetriever' node would always output a single
                       edge with a list of documents. It can be represented as ["ElasticsearchRetriever"].

                       In cases when the predecessor node has multiple outputs, e.g., a "QueryClassifier", the output
                       must be specified explicitly as "QueryClassifier.output_2".
        """
        self.pipeline.add_node(component=component, name=name, inputs=inputs)

    def get_node(self, name: str):
        """
        Get a node from the Pipeline.

        :param name: The name of the node.
        """
        component = self.pipeline.get_node(name)
        return component

    def set_node(self, name: str, component):
        """
        Set the component for a node in the Pipeline.

        :param name: The name of the node.
        :param component: The component object to be set at the node.
        """
        self.pipeline.set_node(name, component)

    def draw(self, path: Path = Path("pipeline.png")):
        """
        Create a Graphviz visualization of the pipeline.

        :param path: the path to save the image.
        """
        self.pipeline.draw(path)


class ExtractiveQAPipeline(BaseStandardPipeline):
    """
    Pipeline for Extractive Question Answering.
    """
    def __init__(self, reader: BaseReader, retriever: BaseRetriever):
        """
        :param reader: Reader instance
        :param retriever: Retriever instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])

    def run(self,
            query: str,
            params: Optional[dict] = None,
            debug: Optional[bool] = None):
        """
        :param query: The search query string.
        :param params: Params for the `retriever` and `reader`. For instance,
                       params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
        :param debug: Whether the pipeline should instruct nodes to collect debug information
                      about their execution. By default these include the input parameters
                      they received and the output they generated. 
                      All debug information can then be found in the dict returned
                      by this method under the key "_debug"
        """
        output = self.pipeline.run(query=query, params=params, debug=debug)
        return output

    def eval(self,
            queries: List[str],
            labels: List[MultiLabel],
            params: Optional[dict]) -> EvaluationResult:
            
        """
        Evaluates the pipeline by running the pipeline once per query in debug mode 
        and putting together all data that is needed for evaluation, e.g. calculating metrics.

        :param queries: The queries to evaluate
        :param labels: The labels to evaluate on
        :param params: Params for the `retriever` and `reader`. For instance,
                       params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
        """
        output = self.pipeline.eval(queries=queries, labels=labels, params=params)
        return output

class DocumentSearchPipeline(BaseStandardPipeline):
    """
    Pipeline for semantic document search.
    """
    def __init__(self, retriever: BaseRetriever):
        """
        :param retriever: Retriever instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])

    def run(self,
            query: str,
            params: Optional[dict] = None,
            debug: Optional[bool] = None):
        """
        :param query: the query string.
        :param params: params for the `retriever` and `reader`. For instance, params={"Retriever": {"top_k": 10}}
        :param debug: Whether the pipeline should instruct nodes to collect debug information
              about their execution. By default these include the input parameters
              they received and the output they generated.
              All debug information can then be found in the dict returned
              by this method under the key "_debug"
        """
        output = self.pipeline.run(query=query, params=params, debug=debug)
        return output


class GenerativeQAPipeline(BaseStandardPipeline):
    """
    Pipeline for Generative Question Answering.
    """
    def __init__(self, generator: BaseGenerator, retriever: BaseRetriever):
        """
        :param generator: Generator instance
        :param retriever: Retriever instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=generator, name="Generator", inputs=["Retriever"])

    def run(self,
            query: str,
            params: Optional[dict] = None,
            debug: Optional[bool] = None):
        """
        :param query: the query string.
        :param params: params for the `retriever` and `generator`. For instance,
                       params={"Retriever": {"top_k": 10}, "Generator": {"top_k": 5}}
        :param debug: Whether the pipeline should instruct nodes to collect debug information
              about their execution. By default these include the input parameters
              they received and the output they generated.
              All debug information can then be found in the dict returned
              by this method under the key "_debug"
        """
        output = self.pipeline.run(query=query, params=params, debug=debug)
        return output


class SearchSummarizationPipeline(BaseStandardPipeline):
    """
    Pipeline that retrieves documents for a query and then summarizes those documents.
    """
    def __init__(self, summarizer: BaseSummarizer, retriever: BaseRetriever, return_in_answer_format: bool = False):
        """
        :param summarizer: Summarizer instance
        :param retriever: Retriever instance
        :param return_in_answer_format: Whether the results should be returned as documents (False) or in the answer
                                        format used in other QA pipelines (True). With the latter, you can use this
                                        pipeline as a "drop-in replacement" for other QA pipelines.
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=summarizer, name="Summarizer", inputs=["Retriever"])
        self.return_in_answer_format = return_in_answer_format

    def run(self,
            query: str,
            params: Optional[dict] = None,
            debug: Optional[bool] = None):
        """
        :param query: the query string.
        :param params: params for the `retriever` and `summarizer`. For instance,
                       params={"Retriever": {"top_k": 10}, "Summarizer": {"generate_single_summary": True}}
        :param debug: Whether the pipeline should instruct nodes to collect debug information
              about their execution. By default these include the input parameters
              they received and the output they generated.
              All debug information can then be found in the dict returned
              by this method under the key "_debug"
        """
        output = self.pipeline.run(query=query, params=params, debug=debug)

        # Convert to answer format to allow "drop-in replacement" for other QA pipelines
        if self.return_in_answer_format:
            results: Dict = {"query": query, "answers": []}
            docs = deepcopy(output["documents"])
            for doc in docs:
                cur_answer = {
                    "query": query,
                    "answer": doc.content,
                    "document_id": doc.id,
                    "context": doc.meta.pop("context"),
                    "score": None,
                    "offset_start": None,
                    "offset_end": None,
                    "meta": doc.meta,
                }

                results["answers"].append(cur_answer)
        else:
            results = output
        return results


class FAQPipeline(BaseStandardPipeline):
    """
    Pipeline for finding similar FAQs using semantic document search.
    """
    def __init__(self, retriever: BaseRetriever):
        """
        :param retriever: Retriever instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=Docs2Answers(), name="Docs2Answers", inputs=["Retriever"])

    def run(self,
            query: str,
            params: Optional[dict] = None,
            debug: Optional[bool] = None):
        """
        :param query: the query string.
        :param params: params for the `retriever`. For instance, params={"Retriever": {"top_k": 10}}
        :param debug: Whether the pipeline should instruct nodes to collect debug information
              about their execution. By default these include the input parameters
              they received and the output they generated.
              All debug information can then be found in the dict returned
              by this method under the key "_debug"
        """
        output = self.pipeline.run(query=query, params=params, debug=debug)
        return output


class TranslationWrapperPipeline(BaseStandardPipeline):
    """
    Takes an existing search pipeline and adds one "input translation node" after the Query and one
    "output translation" node just before returning the results
    """
    def __init__(
        self,
        input_translator: BaseTranslator,
        output_translator: BaseTranslator,
        pipeline: BaseStandardPipeline
    ):
        """
        Wrap a given `pipeline` with the `input_translator` and `output_translator`.

        :param input_translator: A Translator node that shall translate the input query from language A to B
        :param output_translator: A Translator node that shall translate the pipeline results from language B to A
        :param pipeline: The pipeline object (e.g. ExtractiveQAPipeline) you want to "wrap".
                         Note that pipelines with split or merge nodes are currently not supported.
        """

        self.pipeline = Pipeline()
        self.pipeline.add_node(component=input_translator, name="InputTranslator", inputs=["Query"])

        graph = pipeline.pipeline.graph
        previous_node_name = ["InputTranslator"]
        # Traverse in BFS
        for node in graph.nodes:
            if node == "Query":
                continue

            # TODO: Do not work properly for Join Node and Answer format
            if graph.nodes[node]["inputs"] and len(graph.nodes[node]["inputs"]) > 1:
                raise AttributeError("Split and merge nodes are not supported currently")

            self.pipeline.add_node(name=node, component=graph.nodes[node]["component"], inputs=previous_node_name)
            previous_node_name = [node]

        self.pipeline.add_node(component=output_translator, name="OutputTranslator", inputs=previous_node_name)

    def run(self, **kwargs):
        output = self.pipeline.run(**kwargs)
        return output


class QuestionGenerationPipeline(BaseStandardPipeline):
    """
    A simple pipeline that takes documents as input and generates
    questions that it thinks can be answered by the documents.
    """
    def __init__(self, question_generator: QuestionGenerator):
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=question_generator, name="QuestionGenerator", inputs=["Query"])

    def run(self,
            documents,
            params: Optional[dict] = None,
            debug: Optional[bool] = None):
        output = self.pipeline.run(documents=documents, params=params, debug=debug)
        return output


class RetrieverQuestionGenerationPipeline(BaseStandardPipeline):
    """
    A simple pipeline that takes a query as input, performs retrieval, and then generates
    questions that it thinks can be answered by the retrieved documents.
    """
    def __init__(self, retriever: BaseRetriever, question_generator: QuestionGenerator):
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=question_generator, name="Question Generator", inputs=["Retriever"])

    def run(self,
            query: str,
            params: Optional[dict] = None,
            debug: Optional[bool] = None):
        output = self.pipeline.run(query=query, params=params, debug=debug)
        return output


class QuestionAnswerGenerationPipeline(BaseStandardPipeline):
    """
    This is a pipeline which takes a document as input, generates questions that the model thinks can be answered by
    this document, and then performs question answering of this questions using that single document.
    """
    def __init__(self, question_generator: QuestionGenerator, reader: BaseReader):
        setattr(question_generator, "run", self.formatting_wrapper(question_generator.run))
        # Overwrite reader.run function so it can handle a batch of questions being passed on by the QuestionGenerator
        setattr(reader, "run", reader.run_batch)
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=question_generator, name="QuestionGenerator", inputs=["Query"])
        self.pipeline.add_node(component=reader, name="Reader", inputs=["QuestionGenerator"])

    # This is used to format the output of the QuestionGenerator so that its questions are ready to be answered by the reader
    def formatting_wrapper(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            output, output_stream = fn(*args, **kwargs)
            questions = output["generated_questions"][0]["questions"]
            documents = output["documents"]
            query_doc_list = []
            for q in questions:
                query_doc_list.append({"queries": q, "docs": documents})
            kwargs["query_doc_list"] = query_doc_list
            return kwargs, output_stream
        return wrapper

    def run(self,
            documents: List[Document], # type: ignore
            params: Optional[dict] = None,
            debug: Optional[bool] = None):
        output = self.pipeline.run(documents=documents, params=params, debug=debug)
        return output


class MostSimilarDocumentsPipeline(BaseStandardPipeline):
    def __init__(self, document_store: BaseDocumentStore):
        """
        Initialize a Pipeline for finding the most similar documents to a given document.
        This pipeline can be helpful if you already show a relevant document to your end users and they want to search for just similar ones.  

        :param document_store: Document Store instance with already stored embeddings. 
        """
        self.document_store = document_store

    def run(self, document_ids: List[str], top_k: int = 5):
        """
        :param document_ids: document ids
        :param top_k: How many documents id to return against single document
        """
        similar_documents: list = []
        self.document_store.return_embedding = True  # type: ignore

        for document in self.document_store.get_documents_by_id(ids=document_ids):
            similar_documents.append(self.document_store.query_by_embedding(query_emb=document.embedding,
                                                                            return_embedding=False,
                                                                            top_k=top_k))

        self.document_store.return_embedding = False  # type: ignore
        return similar_documents
