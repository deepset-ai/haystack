from typing import Optional, List, Dict, Any, Tuple

from haystack.nodes.base import BaseComponent
from haystack.schema import Document, MultiLabel


def expand_value_to_list(value: Any, target_list: List[Any]) -> Tuple[List[Any]]:
    """
    Transforms a value into a list of the same values, as long as the target list.

    Example:

    ```python
    assert expand_value_to_list(value=1, target_list=list(range(5))) == [1, 1, 1, 1, 1]
    ```
    """
    return ([value] * len(target_list),)


def join_documents(documents: List[Document], delimiter: str) -> Tuple[List[Any]]:
    """
    Transforms a list of documents into a list containing a single Document, whose content
    is the content of all original documents separated by the given delimiter.

    All metadata is dropped. (TODO: fix)

    Example:

    ```python
    assert join_documents(
        documents=[
            Document(content="first"),
            Document(content="second"),
            Document(content="third")
        ],
        separator=" - "
    ) == [Document(content="first - second - third")]
    ```
    """
    return ([Document(content=delimiter.join([d.content for d in documents]))],)


REGISTERED_FUNCTIONS = {"expand_value_to_list": expand_value_to_list, "join_documents": join_documents}


class InvocationContextMapper(BaseComponent):

    """
    InvocationContextMapper is a component that can invoke arbitrary, registered functions, on the invocation context
    (query, documents etc.) of a pipeline and pass the new/modified variables further down the pipeline.

    Using YAML configuration InvocationContextMapper component is initialized with functions to invoke on pipeline invocation
    context.

    For example, in the YAML snippet below:
    ```yaml
        components:
        - name: mapper
          type: InvocationContextMapper
          params:
            func: expand_value_to_list
            inputs:
                value: query
                target_list: documents
            output: [questions]
    ```
    InvocationContextMapper component is initialized with a directive to invoke function expand on the variable query and to store
    the result in the invocation context variable questions. All other invocation context variables are passed down
    the pipeline as is.

    InvocationContextMapper is especially useful in the context of pipelines with PromptNode(s) where we need to modify the invocation
    context to match the templates of PromptNodes.

    Multiple InvocationContextMapper components can be used in a pipeline to modify the invocation context as needed.
    """

    outgoing_edges = 1

    def __init__(self, inputs: Dict[str, str], outputs: List[str], func: str, params: Optional[Dict[str, Any]] = None):
        """
        Initializes a InvocationContextMapper component.

        :param inputs: A dictionary of input parameters for the InvocationContextMapper component. These directives
        are a dictionary version of YAML directives that specify the functions to invoke on the invocation context.

        For example, in the Python snippet below:
        ```python
            mapper = InvocationContextMapper(inputs={"query": {"output": "questions"}})
            pipeline = Pipeline()
            pipeline.add_node(component=InvocationContextMapper, name="mapper", inputs=["Query"])
            ...
        ```
        InvocationContextMapper component is initialized with a directive to rename the invocation context variable query to questions.
        """
        super().__init__()
        self.inputs = inputs
        self.params = params or {}
        self.outputs = outputs
        self.function = REGISTERED_FUNCTIONS[func]

    def run(
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
        invocation_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict, str]:

        # The invocation context overwrites locals(), so if for example invocation_context contains a
        # modified list of Documents under the `documents` key, such list is used.
        invocation_context = {**locals(), **(invocation_context or {})}

        input_values = {key: invocation_context[value] for key, value in self.inputs.items()}

        output_values = self.function(**input_values, **self.params)
        for output_key, output_value in zip(self.outputs, output_values):
            invocation_context[output_key] = output_value

        output = {"invocation_context": invocation_context}
        if query:
            output["query"] = query
        if file_paths:
            output["file_paths"] = file_paths
        if labels:
            output["labels"] = labels
        if documents:
            output["documents"] = documents
        if meta:
            output["meta"] = meta

        return output, "output_1"

    def run_batch(
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
        invocation_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict, str]:

        return self.run(
            query=query,
            file_paths=file_paths,
            labels=labels,
            documents=documents,
            meta=meta,
            invocation_context=invocation_context,
        )
