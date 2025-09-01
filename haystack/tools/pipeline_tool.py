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
    - Automatic LLM tool calling schema generation from component input sockets
    - Automatic name generation from component class name
    - Description extraction from component docstrings

    To use PipelineTool, you first need a Haystack pipeline.
    Below is an example of creating a ComponentTool from an existing SerperDevWebSearch component.

    ## Usage Example:

    ```python
    from haystack import Pipeline
    from haystack.tools import PipelineTool
    from haystack.components.embedders import SentenceTransformerTextEmbedder
    from haystack.components.retrievers import InMemoryEmbeddingRetriever
    from haystack.utils import Secret
    from haystack.components.tools.tool_invoker import ToolInvoker
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    # Create a TextEmbedder and an InMemoryEmbeddingRetriever components
    text_embedder = SentenceTransformerTextEmbedder(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")
    retriever = InMemoryEmbeddingRetriever()

    # Create a tool based on a retrieval pipeline
    tool = ComponentTool(
        component=search,
        name="web_search",  # Optional: defaults to "serper_dev_web_search"
        description="Search the web for current information on any topic"  # Optional: defaults to component docstring
    )

    # Create pipeline with OpenAIChatGenerator and ToolInvoker
    pipeline = Pipeline()
    pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=[tool]))
    pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))

    # Connect components
    pipeline.connect("llm.replies", "tool_invoker.messages")

    message = ChatMessage.from_user("Use the web search tool to find information about Nikola Tesla")

    # Run pipeline
    result = pipeline.run({"llm": {"messages": [message]}})

    print(result)
    ```

    """

    def __init__(
        self,
        pipeline: Pipeline,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[dict[str, Any]] = None,
        *,
        outputs_to_string: Optional[dict[str, Union[str, Callable[[Any], str]]]] = None,
        inputs_from_state: Optional[dict[str, str]] = None,
        outputs_to_state: Optional[dict[str, dict[str, Union[str, Callable]]]] = None,
        input_mapping: Optional[dict[str, str]] = None,
        output_mapping: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Create a Tool instance from a Haystack pipeline.

        :param pipeline: The Haystack pipeline to wrap as a tool.
        :param name: Optional name for the tool (defaults to snake_case of component class name).
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
        :param input_mapping: A dictionary mapping component input names to pipeline input socket paths.
            If not provided, a default input mapping will be created based on all pipeline inputs.
        :param output_mapping: A dictionary mapping pipeline output socket paths to component output names.
            If not provided, a default output mapping will be created based on all pipeline outputs.
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
        """
        serialized: dict[str, Any] = {
            "pipeline": self._pipeline.to_dict(),
            "name": self.name,
            "description": self.description,
            "parameters": self._unresolved_parameters,
            "inputs_from_state": self.inputs_from_state,
            # This is soft-copied as to not modify the attributes in place
            "outputs_to_state": self.outputs_to_state.copy() if self.outputs_to_state else None,
            "input_mapping": self._input_mapping,
            "output_mapping": self._output_mapping,
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
        Deserializes the ComponentTool from a dictionary.
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

        return cls(
            pipeline=pipeline,
            name=inner_data["name"],
            description=inner_data["description"],
            parameters=inner_data.get("parameters", None),
            outputs_to_string=inner_data.get("outputs_to_string", None),
            inputs_from_state=inner_data.get("inputs_from_state", None),
            outputs_to_state=inner_data.get("outputs_to_state", None),
        )
