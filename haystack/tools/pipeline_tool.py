# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable

from haystack import AsyncPipeline, Pipeline, SuperComponent, logging
from haystack.core.serialization import generate_qualified_class_name
from haystack.tools.component_tool import ComponentTool
from haystack.tools.tool import _deserialize_outputs_to_state, _serialize_outputs_to_state
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
    from haystack.dataclasses import ChatMessage
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.components.embedders.sentence_transformers_text_embedder import SentenceTransformersTextEmbedder
    from haystack.components.embedders.sentence_transformers_document_embedder import (
        SentenceTransformersDocumentEmbedder
    )
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.components.retrievers import InMemoryEmbeddingRetriever
    from haystack.components.agents import Agent
    from haystack.tools import PipelineTool

    # Initialize a document store and add some documents
    document_store = InMemoryDocumentStore()
    document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    documents = [
        Document(content="Nikola Tesla was a Serbian-American inventor and electrical engineer."),
        Document(
            content="He is best known for his contributions to the design of the modern alternating current (AC) "
                    "electricity supply system."
        ),
    ]
    document_embedder.warm_up()
    docs_with_embeddings = document_embedder.run(documents=documents)["documents"]
    document_store.write_documents(docs_with_embeddings)

    # Build a simple retrieval pipeline
    retrieval_pipeline = Pipeline()
    retrieval_pipeline.add_component(
        "embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    )
    retrieval_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))

    retrieval_pipeline.connect("embedder.embedding", "retriever.query_embedding")

    # Wrap the pipeline as a tool
    retriever_tool = PipelineTool(
        pipeline=retrieval_pipeline,
        input_mapping={"query": ["embedder.text"]},
        output_mapping={"retriever.documents": "documents"},
        name="document_retriever",
        description="For any questions about Nikola Tesla, always use this tool",
    )

    # Create an Agent with the tool
    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-4.1-mini"),
        tools=[retriever_tool]
    )

    # Let the Agent handle a query
    result = agent.run([ChatMessage.from_user("Who was Nikola Tesla?")])

    # Print result of the tool call
    print("Tool Call Result:")
    print(result["messages"][2].tool_call_result.result)
    print("")

    # Print answer
    print("Answer:")
    print(result["messages"][-1].text)
    ```
    """

    def __init__(
        self,
        pipeline: Pipeline | AsyncPipeline,
        *,
        name: str,
        description: str,
        input_mapping: dict[str, list[str]] | None = None,
        output_mapping: dict[str, str] | None = None,
        parameters: dict[str, Any] | None = None,
        outputs_to_string: dict[str, str | Callable[[Any], str]] | None = None,
        inputs_from_state: dict[str, str] | None = None,
        outputs_to_state: dict[str, dict[str, str | Callable]] | None = None,
        outputs_to_result: dict[str, Any] | None = None,
    ) -> None:
        """
        Create a Tool instance from a Haystack pipeline.

        :param pipeline: The Haystack pipeline to wrap as a tool.
        :param name: Name of the tool.
        :param description: Description of the tool.
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
            ```
        :param parameters:
            A JSON schema defining the parameters expected by the Tool.
            Will fall back to the parameters defined in the component's run method signature if not provided.
        :param outputs_to_string:
            Optional dictionary defining how tool outputs should be converted into string(s).
            If not provided, the tool result is converted to a string using a default handler.
            If you want to return the tool result without string conversion, use `outputs_to_result`.

            `outputs_to_string` supports two formats:

            1. Single output format - use "source" and/or "handler" at the root level:
            ```python
            {
                "source": "docs", "handler": format_documents
            }
            ```
            If the source is provided, only the specified output key is sent to the handler.
            If the source is omitted, the whole tool result is sent to the handler.

            2. Multiple output format - map keys to individual configurations:
            ```python
            {
                "formatted_docs": {"source": "docs", "handler": format_documents},
                "summary": {"source": "summary_text", "handler": str.upper}
            }
            ```
            Each key maps to a dictionary that can contain "source" and/or "handler".
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
        :param outputs_to_result:
            Optional dictionary defining how the tool output is converted into a result. This is an alternative to
            `outputs_to_string` and supports returning images.

            The configuration dictionary can contain:
            - `source`: If the tool output is a dictionary, extract the value of this key.
            - `handler`: A function that takes the tool output (or the extracted source value) and returns the
                final result. The handler should typically return a list of `TextContent`/`ImageContent` objects.

            Behaviors:
            - `{}`: Returns the raw tool output as is.
            - `{"handler": my_func}`: Passes the full tool output to `my_func`.
            - `{"source": "my_key"}`: If the output is a dictionary, returns `output["my_key"]`.
            - `{"source": "my_key", "handler": my_func}`: If the output is a dictionary, passes `output["my_key"]` to
                `my_func`.
        :raises ValueError: If the provided pipeline is not a valid Haystack Pipeline instance.
        """
        if not isinstance(pipeline, (Pipeline, AsyncPipeline)):
            raise ValueError(
                "The 'pipeline' parameter must be an instance of Pipeline or AsyncPipeline."
                f" Got {type(pipeline)} instead."
            )

        super().__init__(
            component=SuperComponent(pipeline=pipeline, input_mapping=input_mapping, output_mapping=output_mapping),
            name=name,
            description=description,
            parameters=parameters,
            outputs_to_string=outputs_to_string,
            inputs_from_state=inputs_from_state,
            outputs_to_state=outputs_to_state,
            outputs_to_result=outputs_to_result,
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
            "is_pipeline_async": isinstance(self._pipeline, AsyncPipeline),
            "outputs_to_state": _serialize_outputs_to_state(self.outputs_to_state) if self.outputs_to_state else None,
            "outputs_to_result": self.outputs_to_result,
        }

        if self.outputs_to_string is not None and self.outputs_to_string.get("handler") is not None:
            # This is soft-copied as to not modify the attributes in place
            serialized["outputs_to_string"] = self.outputs_to_string.copy()
            serialized["outputs_to_string"]["handler"] = serialize_callable(self.outputs_to_string["handler"])
        else:
            serialized["outputs_to_string"] = None

        if serialized["outputs_to_result"] is not None and serialized["outputs_to_result"].get("handler") is not None:
            serialized["outputs_to_result"]["handler"] = serialize_callable(serialized["outputs_to_result"]["handler"])

        return {"type": generate_qualified_class_name(type(self)), "data": serialized}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineTool":
        """
        Deserializes the PipelineTool from a dictionary.

        :param data: The dictionary representation of PipelineTool.
        :returns:
            The deserialized PipelineTool instance.
        """
        inner_data = data["data"]
        is_pipeline_async = inner_data.get("is_pipeline_async", False)
        pipeline_class = AsyncPipeline if is_pipeline_async else Pipeline
        pipeline = pipeline_class.from_dict(inner_data["pipeline"])

        if "outputs_to_state" in inner_data and inner_data["outputs_to_state"]:
            inner_data["outputs_to_state"] = _deserialize_outputs_to_state(inner_data["outputs_to_state"])

        if (
            inner_data.get("outputs_to_string") is not None
            and inner_data["outputs_to_string"].get("handler") is not None
        ):
            inner_data["outputs_to_string"]["handler"] = deserialize_callable(
                inner_data["outputs_to_string"]["handler"]
            )

        if (
            inner_data.get("outputs_to_result") is not None
            and inner_data["outputs_to_result"].get("handler") is not None
        ):
            inner_data["outputs_to_result"]["handler"] = deserialize_callable(
                inner_data["outputs_to_result"]["handler"]
            )

        merged_data = {**inner_data, "pipeline": pipeline}
        # Remove is_pipeline_async as it's not a parameter of the constructor
        merged_data.pop("is_pipeline_async", None)
        return cls(**merged_data)
