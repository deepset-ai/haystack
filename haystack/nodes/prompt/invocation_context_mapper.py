from typing import Optional, List, Dict, Any, Tuple, Union

from haystack.nodes.base import BaseComponent
from haystack.schema import Document, MultiLabel


def expand_value_to_list(value: Any, target_list: List[Any]) -> Tuple[List[Any]]:
    """
    Transforms a value into a list of the same values, as long as the target list.

    Example:

    ```python
    assert expand_value_to_list(value=1, target_list=list(range(5))) == ([1, 1, 1, 1, 1], )
    ```
    """
    return ([value] * len(target_list),)


def join_strings(strings: List[str], delimiter: str = " ") -> Tuple[List[Any]]:
    """
    Transforms a list of strings into a list containing a single string, whose content
    is the content of all original strings separated by the given delimiter.

    Example:

    ```python
    assert join_strings([strings="first", "second", "third"], separator=" - ") == (["first - second - third"], )
    ```
    """
    return ([delimiter.join([d.content for d in strings])],)


def join_documents(documents: List[Document], delimiter: str = " ") -> Tuple[List[Any]]:
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
    ) == ([Document(content="first - second - third")], )
    ```
    """
    return ([Document(content=delimiter.join([d.content for d in documents]))],)


def convert_to_docs(
    strings: List[str], metadata: Optional[Union[List[str], str]] = None, id_hash_keys: Optional[List[str]] = None
) -> Tuple[List[Any]]:
    """
    Transforms a list of strings into a list of Documents. If the metadata is given as a single
    dict, all documents will receive the same meta; if the metadata is given as a list, such list
    must be as long as the list of strings and each Document will receive its own metadata.
    On the contrary, id_hash_keys can only be given once and it will be assigned to all documents.

    Example:

    ```python
    assert join_strings(
            strings=["first", "second", "third"],
            metadata=[{"position": i} for i in range(3)],
            id_hash_keys=['content', 'meta]
        ) == [
            Document(content="first", metadata={"position": 1}, id_hash_keys=['content', 'meta])]),
            Document(content="second", metadata={"position": 2}, id_hash_keys=['content', 'meta]),
            Document(content="third", metadata={"position": 3}, id_hash_keys=['content', 'meta])
        ]
    ```
    """

    if isinstance(metadata, dict):
        all_metadata = [metadata]
    elif isinstance(metadata, list):
        if len(metadata) != len(strings):
            raise ValueError(
                f"Not enough metadata dictionaries. convert_to_docs received {len(strings)} and {len(metadata)} metadata dictionaries."
            )
        all_metadata = metadata

    return (
        [Document(content=string, meta=meta, id_hash_keys=id_hash_keys) for string, meta in zip(strings, all_metadata)],
    )


REGISTERED_FUNCTIONS = {
    "expand_value_to_list": expand_value_to_list,
    "join_strings": join_strings,
    "join_documents": join_documents,
    "convert_to_docs": convert_to_docs,
}


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

    - len: parameters are target:Any; returns int; built-in len function returning the length of the input variable
    - expand: parameters are [expand_target:Any, size:int]; returns List[Any]; it returns a list of specified size
        for the input variable
    - concat: parameters are [texts:List[str], delimiter:str]; returns str, concatenates texts with the specified
        delimiter
    - concat_docs: parameters are [docs: List[Document]], delimiter:str]; returns str; concatenates the docs with the
        specified delimiter and returns a string
    - convert_to_docs: parameters are texts: List[str]; returns List[Document]; converts the input list of strings
        to a list of Documents
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
        invocation_context.pop("self")

        try:
            input_values = {key: invocation_context[value] for key, value in self.inputs.items()}
        except KeyError as e:
            missing_values = [value for value in self.inputs.values() if value not in invocation_context.keys()]
            raise ValueError(
                f"InvocationContextMapper could not find these values from your inputs list in the invocation context: {missing_values}. "
                "Make sure the value exists in the invocation context."
            ) from e

        try:
            output_values = self.function(**input_values, **self.params)
        except TypeError as e:
            raise ValueError(
                "InvocationContextMapper could not apply the function to your inputs and parameters. "
                "Check the above stacktrace and make sure you provided all the correct inputs, parameters, "
                "and parameter types."
            ) from e

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
