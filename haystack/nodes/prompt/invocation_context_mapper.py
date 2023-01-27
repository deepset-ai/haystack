from typing import Optional, List, Dict, Any, Tuple, Union, Callable

import logging

from haystack.nodes.base import BaseComponent
from haystack.schema import Document, MultiLabel


logger = logging.getLogger(__name__)


def rename(value: Any) -> Tuple[Any]:
    """
    Identity function. Can be used to rename values in the invocation context without changing them.

    Example:

    ```python
    assert rename(1) == (1, )
    ```
    """
    return (value,)


def expand_value_to_list(value: Any, target_list: List[Any]) -> Tuple[List[Any]]:
    """
    Transforms a value into a list containing this value as many times as the length of the target list.

    Example:

    ```python
    assert expand_value_to_list(value=1, target_list=list(range(5))) == ([1, 1, 1, 1, 1], )
    ```
    """
    return ([value] * len(target_list),)


def join_lists(lists: List[List[Any]]) -> Tuple[List[Any]]:
    """
    Joins the passed lists into a single one.

    Example:

    ```python
    assert join_lists(lists=[[1, 2, 3], [4, 5]]) == ([1, 2, 3, 4, 5], )
    ```
    """
    merged_list = []
    for inner_list in lists:
        merged_list += inner_list
    return (merged_list,)


def join_strings(strings: List[str], delimiter: str = " ") -> Tuple[List[str]]:
    """
    Transforms a list of strings into a list containing a single string. The content of this list
    is the content of all original strings separated by the delimiter you specify.

    Example:

    ```python
    assert join_strings(strings=["first", "second", "third"], separator=" - ") == (["first - second - third"], )
    ```
    """
    return ([delimiter.join(strings)],)


def join_documents(documents: List[Document], delimiter: str = " ") -> Tuple[List[Document]]:
    """
    Transforms a list of documents into a list containing a single Document. The content of this list
    is the content of all original documents separated by the delimiter you specify.

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


def strings_to_documents(
    strings: List[str],
    meta: Union[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] = None,
    id_hash_keys: Optional[List[str]] = None,
) -> Tuple[List[Document]]:
    """
    Transforms a list of strings into a list of Documents. If you pass the metadata a single
    dictionary, all Documents get the same metadata. If you pass the metadata as a list, the length of this list
    must the same as the length of the list of strings and each Document gets its own metadata.
    You can specify `id_hash_keys` only once and it gets assigned to all Documents.

    Example:

    ```python
    assert strings_to_documents(
            strings=["first", "second", "third"],
            meta=[{"position": i} for i in range(3)],
            id_hash_keys=['content', 'meta]
        ) == [
            Document(content="first", metadata={"position": 1}, id_hash_keys=['content', 'meta])]),
            Document(content="second", metadata={"position": 2}, id_hash_keys=['content', 'meta]),
            Document(content="third", metadata={"position": 3}, id_hash_keys=['content', 'meta])
        ]
    ```
    """
    all_metadata: List[Optional[Dict[str, Any]]]
    if isinstance(meta, dict):
        all_metadata = [meta] * len(strings)
    elif isinstance(meta, list):
        if len(meta) != len(strings):
            raise ValueError(
                f"Not enough metadata dictionaries. strings_to_documents received {len(strings)} and {len(meta)} metadata dictionaries."
            )
        all_metadata = meta
    else:
        all_metadata = [None] * len(strings)

    return ([Document(content=string, meta=m, id_hash_keys=id_hash_keys) for string, m in zip(strings, all_metadata)],)


def documents_to_strings(documents: List[Document]) -> Tuple[List[str]]:
    """
    Extracts the content field of Documents and returns a list of strings.

    Example:

    ```python
    assert documents_to_strings(
            documents=[
                Document(content="first"),
                Document(content="second"),
                Document(content="third")
            ]
        ) == ["first", "second", "third"],
    ```
    """
    return ([doc.content for doc in documents],)


REGISTERED_FUNCTIONS: Dict[str, Callable[..., Tuple[Any]]] = {
    "rename": rename,
    "expand_value_to_list": expand_value_to_list,
    "join_lists": join_lists,
    "join_strings": join_strings,
    "join_documents": join_documents,
    "strings_to_documents": strings_to_documents,
    "documents_to_strings": documents_to_strings,
}


class InvocationContextMapper(BaseComponent):

    """
    InvocationContextMapper is a component that can invoke arbitrary, registered functions on the invocation context
    (query, documents, and so on) of a pipeline. It then passes the new or modified variables further down the pipeline.

    Using YAML configuration, the InvocationContextMapper component is initialized with functions to invoke on pipeline invocation
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
    the pipeline as they are.

    InvocationContextMapper is especially useful for pipelines with PromptNodes, where we need to modify the invocation
    context to match the templates of PromptNodes.

    You can use multiple InvocationContextMapper components in a pipeline to modify the invocation context as needed.

    `InvocationContextMapper` supports the current functions:

    - `expand_value_to_list`
    - `join_strings`
    - `join_documents`
    - `join_lists`
    - `strings_to_documents`
    - `documents_to_strings`

    See their descriptions in the code for details about their inputs, outputs, and other parameters.
    """

    outgoing_edges = 1

    def __init__(
        self,
        func: str,
        outputs: List[str],
        inputs: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the InvocationContextMapper component.

        Some examples:

        ```yaml
        - name: mapper
          type: InvocationContextMapper
          params:
          func: expand_value_to_list
          inputs:
            value: query
            target_list: documents
          outputs:
            - questions
        ```
        This node takes the content of `query` and creates a list that contains the value of `query` `len(documents)` times.
        This list is stored in the invocation context under the key `questions`.

        ```yaml
        - name: mapper
          type: InvocationContextMapper
          params:
          func: join_documents
          inputs:
            value: documents
          params:
            delimiter: ' - '
          outputs:
            - documents
        ```
        This node overwrites the content of `documents` in the invocation context with a list containing a single Document
        whose content is the concatenation of all the original Documents. So if `documents` contained
        `[Document("A"), Document("B"), Document("C")]`, this mapper overwrites it with `[Document("A - B - C")]`

        ```yaml
        - name: mapper
          type: InvocationContextMapper
          params:
          func: join_strings
          params:
            strings: ['a', 'b', 'c']
            delimiter: ' . '
          outputs:
            - single_string

        - name: mapper
          type: InvocationContextMapper
          params:
          func: strings_to_documents
          inputs:
            strings: single_string
            metadata:
              name: 'my_file.txt'
          outputs:
            - single_document
        ```
        These two nodes, executed one after the other, first add a key in the invocation context called `single_string`
        that contains `a . b . c`, and then create another key called `single_document` that contains instead
        `[Document(content="a . b . c", metadata={'name': 'my_file.txt'})]`.

        :param func: The function to apply.
        :param inputs: Maps the function's input kwargs to the key-value pairs in the invocation context.
            For example, `expand_value_to_list` expects the `value` and `target_list` parameters, so `inputs` might contain:
            `{'value': 'query', 'target_list': 'documents'}`. It doesn't need to contain all keyword args, see `params`.
        :param params: Maps the function's input kwargs to some fixed values. For example, `expand_value_to_list` expects
            `value` and `target_list` parameters, so `params` might contain
            `{'value': 'A', 'target_list': [1, 1, 1, 1]}` and the node's output is `["A", "A", "A", "A"]`.
            It doesn't need to contain all keyword args, see `inputs`.
            You can use params to provide fallback values for arguments of `run` that you're not sure exist.
            So if you need `query` to exist, you can provide a fallback value in the params, which will be used only if `query`
            is not passed to this node by the pipeline.
        :param outputs: THe key to store the outputs in the invocation context. The length of the outputs must match
            the number of outputs produced by the function invoked.
        """
        super().__init__()
        self.function = REGISTERED_FUNCTIONS[func]
        self.outputs = outputs
        self.inputs = inputs or {}
        self.params = params or {}

    def run(  # type: ignore
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
        invocation_context = {key: value for key, value in invocation_context.items() if value is not None}

        input_values = {
            key: invocation_context[value]
            for key, value in self.inputs.items()
            if value in invocation_context.keys() and value is not None
        }

        input_values = {**self.params, **input_values}
        try:
            logger.debug(
                "InvocationContextMapper is invoking this function: %s(%s)",
                self.function.__name__,
                ", ".join([f"{key}={value}" for key, value in input_values.items()]),
            )
            output_values = self.function(**input_values)
        except TypeError as e:
            raise ValueError(
                "InvocationContextMapper couldn't apply the function to your inputs and parameters. "
                "Check the above stacktrace and make sure you provided all the correct inputs, parameters, "
                "and parameter types."
            ) from e

        for output_key, output_value in zip(self.outputs, output_values):
            invocation_context[output_key] = output_value

        return {"invocation_context": invocation_context}, "output_1"

    def run_batch(  # type: ignore
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
