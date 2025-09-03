# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional, Union

from haystack import Pipeline, SuperComponent, logging
from haystack.core.serialization import generate_qualified_class_name
from haystack.tools.component_tool import ComponentTool
from haystack.tools.tool import Tool, _deserialize_outputs_to_state, _serialize_outputs_to_state
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable

logger = logging.getLogger(__name__)


class PipelineTool(ComponentTool):
    """
    A Tool that wraps Haystack Pipelines, allowing them to be used as tools by LLMs.

    PipelineTool automatically generates LLM-compatible tool schemas from pipeline input sockets,
    which are derived from the underlying components in the pipeline.

    Key features:
    - Automatic LLM tool calling schema generation from pipeline inputs
    - Description extraction of pipeline inputs based on the underlying component docstrings

    To use PipelineTool, you first need a Haystack pipeline.
    Below is an example of creating a PipelineTool

    ## Usage Example:

    ```python
    from haystack import Document, Pipeline
    from haystack.document_stores import InMemoryDocumentStore
    from haystack.components.embedders import SentenceTransformerTextEmbedder, SentenceTransformersDocumentEmbedder
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.components.retrievers import InMemoryEmbeddingRetriever
    from haystack.components.agents import Agent
    from haystack.tools import PipelineTool

    # Initialize a document store and add some documents
    document_store = InMemoryDocumentStore()
    document_embedder = SentenceTransformersDocumentEmbedder(
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"
    )
    documents = [
        Document(content="Nikola Tesla was a Serbian-American inventor and electrical engineer."),
        Document(
            content="He is best known for his contributions to the design of the modern alternating current (AC) "
                    "electricity supply system."
        ),
    ]
    docs_with_embeddings = document_embedder.run(documents=documents)["documents"]
    document_store.write_documents(docs_with_embeddings)

    # Build a simple retrieval pipeline
    retrieval_pipeline = Pipeline()
    retrieval_pipeline.add_component(
        "embedder", SentenceTransformerTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    )
    retrieval_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))

    retrieval_pipeline.connect("embedder.embedding", "retriever.query_embedding")

    # Wrap the pipeline as a tool
    retriever_tool = PipelineTool(
        pipeline=retrieval_pipeline,
        input_mapping={"query": ["embedder.query"]},
        output_mapping={"retriever.documents": "documents"},
        name="document_retriever",
        description="Retrieve documents relevant to a query from the document store",
    )

    # Create an Agent with the tool
    agent = Agent(
        chat_generator=OpenAIChatGenerator(model_name="gpt-4.1-mini"),
        tools=[retriever_tool]
    )

    # Let the Agent handle a query
    result = agent.run("Find information about Nikola Tesla in the document store")

    print(result)
    ```

    """

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        name: str,
        input_mapping: Optional[dict[str, str]] = None,
        output_mapping: Optional[dict[str, str]] = None,
        description: Optional[str] = None,
        parameters: Optional[dict[str, Any]] = None,
        outputs_to_string: Optional[dict[str, Union[str, Callable[[Any], str]]]] = None,
        inputs_from_state: Optional[dict[str, str]] = None,
        outputs_to_state: Optional[dict[str, dict[str, Union[str, Callable]]]] = None,
    ) -> None:
        """
        Create a Tool instance from a Haystack pipeline.

        :param pipeline: The Haystack pipeline to wrap as a tool.
        :param name: Name of the tool.
        :param input_mapping: A dictionary mapping component input names to pipeline input socket paths.
            If not provided, a default input mapping will be created based on all pipeline inputs.
            Example:
            ```python
            input_mapping={
                "query": ["retriever.query", "prompt_builder.query"],
            }
            ```
        :param output_mapping: A dictionary mapping pipeline output socket paths to component output names.
            If not provided, a default output mapping will be created based on all pipeline outputs.
            Example:
            ```python
            output_mapping={
                "retriever.documents": "documents",
                "generator.replies": "replies",
            }
        :param description: Optional description (defaults to component's docstring).
        :param parameters:
            A JSON schema defining the parameters expected by the Tool.
            Will fall back to the parameters defined in the component's run method signature if not provided.
        :param outputs_to_string:
            Optional dictionary defining how a tool outputs should be converted into a string.
            If the source is provided only the specified output key is sent to the handler.
            If the source is omitted the whole tool result is sent to the handler.
            Example:
            ```python
            {
                "source": "docs", "handler": format_documents
            }
            ```
        :param inputs_from_state:
            Optional dictionary mapping state keys to tool parameter names.
            Example: `{"repository": "repo"}` maps state's "repository" to tool's "repo" parameter.
        :param outputs_to_state:
            Optional dictionary defining how tool outputs map to keys within state as well as optional handlers.
            If the source is provided only the specified output key is sent to the handler.
            Example:
            ```python
            {
                "documents": {"source": "docs", "handler": custom_handler}
            }
            ```
            If the source is omitted the whole tool result is sent to the handler.
            Example:
            ```python
            {
                "documents": {"handler": custom_handler}
            }
            ```
        :raises ValueError: If the provided pipeline is not a valid Haystack Pipeline instance.
        """
        if not isinstance(pipeline, Pipeline):
            raise ValueError(
                f"Object {pipeline!r} is not a Haystack pipeline. "
                f"Use PipelineTool only with Haystack component instances."
            )

        super().__init__(
            component=SuperComponent(pipeline=pipeline, input_mapping=input_mapping, output_mapping=output_mapping),
            name=name,
            description=description,
            parameters=parameters,
            outputs_to_string=outputs_to_string,
            inputs_from_state=inputs_from_state,
            outputs_to_state=outputs_to_state,
        )
        self._unresolved_parameters = parameters
        self._pipeline = pipeline
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the PipelineTool to a dictionary.

        :returns:
            The serialized dictionary representation of PipelineTool.
        """
        serialized: dict[str, Any] = {
            "pipeline": self._pipeline.to_dict(),
            "name": self.name,
            "input_mapping": self._input_mapping,
            "output_mapping": self._output_mapping,
            "description": self.description,
            "parameters": self._unresolved_parameters,
            "inputs_from_state": self.inputs_from_state,
            # This is soft-copied as to not modify the attributes in place
            "outputs_to_state": self.outputs_to_state.copy() if self.outputs_to_state else None,
        }

        if self.outputs_to_state is not None:
            serialized["outputs_to_state"] = _serialize_outputs_to_state(self.outputs_to_state)

        if self.outputs_to_string is not None and self.outputs_to_string.get("handler") is not None:
            # This is soft-copied as to not modify the attributes in place
            serialized["outputs_to_string"] = self.outputs_to_string.copy()
            serialized["outputs_to_string"]["handler"] = serialize_callable(self.outputs_to_string["handler"])
        else:
            serialized["outputs_to_string"] = None

        return {"type": generate_qualified_class_name(type(self)), "data": serialized}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tool":
        """
        Deserializes the PipelineTool from a dictionary.

        :param data: The dictionary representation of PipelineTool.
        :returns:
            The deserialized PipelineTool instance.
        """
        inner_data = data["data"]
        pipeline = Pipeline.from_dict(inner_data["pipeline"])

        if "outputs_to_state" in inner_data and inner_data["outputs_to_state"]:
            inner_data["outputs_to_state"] = _deserialize_outputs_to_state(inner_data["outputs_to_state"])

        if (
            inner_data.get("outputs_to_string") is not None
            and inner_data["outputs_to_string"].get("handler") is not None
        ):
            inner_data["outputs_to_string"]["handler"] = deserialize_callable(
                inner_data["outputs_to_string"]["handler"]
            )

        return cls(**{**inner_data, "pipeline": pipeline})
