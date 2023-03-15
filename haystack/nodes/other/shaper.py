from functools import reduce
import re
from string import Template
from typing import Optional, List, Dict, Any, Tuple, Union, Callable

import logging

from haystack.nodes.base import BaseComponent
from haystack.schema import Document, Answer, MultiLabel


logger = logging.getLogger(__name__)


def rename(value: Any) -> Any:
    """
    Identity function. Can be used to rename values in the invocation context without changing them.

    Example:

    ```python
    assert rename(1) == 1
    ```
    """
    return value


def value_to_list(value: Any, target_list: List[Any]) -> List[Any]:
    """
    Transforms a value into a list containing this value as many times as the length of the target list.

    Example:

    ```python
    assert value_to_list(value=1, target_list=list(range(5))) == [1, 1, 1, 1, 1]
    ```
    """
    return [value] * len(target_list)


def join_lists(lists: List[List[Any]]) -> List[Any]:
    """
    Joins the passed lists into a single one.

    Example:

    ```python
    assert join_lists(lists=[[1, 2, 3], [4, 5]]) == [1, 2, 3, 4, 5]
    ```
    """
    merged_list = []
    for inner_list in lists:
        merged_list += inner_list
    return merged_list


def join_strings(strings: List[str], delimiter: str = " ", str_replace: Optional[Dict[str, str]] = None) -> str:
    """
    Transforms a list of strings into a single string. The content of this string
    is the content of all original strings separated by the delimiter you specify.

    Example:

    ```python
    assert join_strings(strings=["first", "second", "third"], delimiter=" - ", str_replace={"r", "R"}) == "fiRst - second - thiRd"
    ```
    """
    str_replace = str_replace or {}
    return delimiter.join([format_string(string, str_replace) for string in strings])


def format_string(string: str, str_replace: Optional[Dict[str, str]] = None) -> str:
    """
    Transforms a string using a substitution dict.

    Example:

    ```python
    assert format_string(string="first", str_replace={"r", "R"}) == "fiRst"
    ```
    """
    str_replace = str_replace or {}
    return reduce(lambda s, kv: s.replace(*kv), str_replace.items(), string)


def join_documents(
    documents: List[Document],
    delimiter: str = " ",
    pattern: Optional[str] = None,
    str_replace: Optional[Dict[str, str]] = None,
) -> List[Document]:
    """
    Transforms a list of documents into a list containing a single Document. The content of this list
    is the joined result of all original documents separated by the delimiter you specify.
    How each document is represented is controlled by the pattern parameter.
    You can use the following placeholders:
    - $content: the content of the document
    - $idx: the index of the document in the list
    - $META_FIELD: the value of the metadata field of name 'META_FIELD'

    All metadata is dropped. (TODO: fix)

    Example:

    ```python
    assert join_documents(
        documents=[
            Document(content="first"),
            Document(content="second"),
            Document(content="third")
        ],
        delimiter=" - ",
        pattern="[$idx] $content",
        str_replace={"r", "R"}
    ) == [Document(content="[1] fiRst - [2] second - [3] thiRd")]
    ```
    """
    return [Document(content=join_documents_to_string(documents, delimiter, pattern, str_replace))]


def format_document(
    document: Document,
    pattern: Optional[str] = None,
    str_replace: Optional[Dict[str, str]] = None,
    idx: Optional[int] = None,
) -> str:
    """
    Transforms a document into a single string.
    How the document is represented is controlled by the pattern parameter.
    You can use the following placeholders:
    - $content: the content of the document
    - $idx: the index of the document in the list
    - $META_FIELD: the value of the metadata field of name 'META_FIELD'

    Example:

    ```python
    assert format_document(
        document=Document(content="first"),
        pattern="prefix [$idx] $content",
        str_replace={"r", "R"},
        idx=1,
    ) == "prefix [1] fiRst"
    ```
    """
    str_replace = str_replace or {}
    pattern = pattern or "$content"

    template = Template(pattern)
    pattern_params = [
        match.groupdict().get("named", match.groupdict().get("braced"))
        for match in template.pattern.finditer(template.template)
    ]
    meta_params = [param for param in pattern_params if param and param not in ["content", "idx"]]
    content = template.substitute(
        {
            "idx": idx,
            "content": reduce(lambda content, kv: content.replace(*kv), str_replace.items(), document.content),
            **{
                k: reduce(lambda val, kv: val.replace(*kv), str_replace.items(), document.meta.get(k, ""))
                for k in meta_params
            },
        }
    )
    return content


def format_answer(
    answer: Answer,
    pattern: Optional[str] = None,
    str_replace: Optional[Dict[str, str]] = None,
    idx: Optional[int] = None,
) -> str:
    """
    Transforms an answer into a single string.
    How the answer is represented is controlled by the pattern parameter.
    You can use the following placeholders:
    - $answer: the answer text of the answer
    - $idx: the index of the answer in the list
    - $META_FIELD: the value of the metadata field of name 'META_FIELD'

    Example:

    ```python
    assert format_answer(
        answer=Answer(answer="first"),
        pattern="prefix [$idx] $answer",
        str_replace={"r", "R"},
        idx=1,
    ) == "prefix [1] fiRst"
    ```
    """
    str_replace = str_replace or {}
    pattern = pattern or "$answer"

    template = Template(pattern)
    pattern_params = [
        match.groupdict().get("named", match.groupdict().get("braced"))
        for match in template.pattern.finditer(template.template)
    ]
    meta_params = [param for param in pattern_params if param and param not in ["answer", "idx"]]
    meta = answer.meta or {}
    content = template.substitute(
        {
            "idx": idx,
            "answer": reduce(lambda content, kv: content.replace(*kv), str_replace.items(), answer.answer),
            **{k: reduce(lambda val, kv: val.replace(*kv), str_replace.items(), meta.get(k, "")) for k in meta_params},
        }
    )
    return content


def join_documents_to_string(
    documents: List[Document],
    delimiter: str = " ",
    pattern: Optional[str] = None,
    str_replace: Optional[Dict[str, str]] = None,
) -> str:
    """
    Transforms a list of documents into a single string. The content of this string
    is the joined result of all original documents separated by the delimiter you specify.
    How each document is represented is controlled by the pattern parameter.
    You can use the following placeholders:
    - $content: the content of the document
    - $idx: the index of the document in the list
    - $META_FIELD: the value of the metadata field of name 'META_FIELD'

    Example:

    ```python
    assert join_documents_to_string(
        documents=[
            Document(content="first"),
            Document(content="second"),
            Document(content="third")
        ],
        delimiter=" - ",
        pattern="[$idx] $content",
        str_replace={"r", "R"}
    ) == "[1] fiRst - [2] second - [3] thiRd"
    ```
    """
    content = delimiter.join(
        format_document(doc, pattern, str_replace, idx=idx) for idx, doc in enumerate(documents, start=1)
    )
    return content


def strings_to_answers(
    strings: List[str],
    prompt: Optional[str] = None,
    documents: Optional[List[Document]] = None,
    document_idx_pattern: Optional[str] = None,
) -> List[Answer]:
    """
    Transforms a list of strings into a list of Answers.

    Example:

    ```python
    assert strings_to_answers(strings=["first", "second", "third"], prompt="prompt", documents=[Document(id="123", content="content")]) == [
            Answer(answer="first", type="generative", document_ids=["123"], meta={"prompt": "prompt"}),
            Answer(answer="second", type="generative", document_ids=["123"], meta={"prompt": "prompt"}),
            Answer(answer="third", type="generative", document_ids=["123"], meta={"prompt": "prompt"}),
        ]
    ```
    """
    candidates = {str(idx): doc.id for idx, doc in enumerate(documents, start=1)} if documents else {}
    answers = []
    for string in strings:
        document_ids = parse_document_ids(
            string=string, document_idx_pattern=document_idx_pattern, candidates=candidates
        )
        answer = Answer(answer=string, type="generative", document_ids=document_ids, meta={"prompt": prompt})
        answers.append(answer)
    return answers


def parse_document_ids(
    string: str, document_idx_pattern: Optional[str] = None, candidates: Optional[Dict[str, str]] = None
) -> Optional[List[str]]:
    """
    Parses an answer string for document indexes and converts them to document ids.
    """
    if not candidates:
        return None
    if not document_idx_pattern:
        return list(candidates.values())

    document_idxs = re.findall(document_idx_pattern, string)
    return [candidates[idx] for idx in document_idxs if idx in candidates]


def answers_to_strings(
    answers: List[Answer], pattern: Optional[str] = None, str_replace: Optional[Dict[str, str]] = None
) -> List[str]:
    """
    Extracts the content field of Documents and returns a list of strings.

    Example:

    ```python
    assert answers_to_strings(
            answers=[
                Answer(answer="first"),
                Answer(answer="second"),
                Answer(answer="third")
            ],
            pattern="[$idx] $answer",
            str_replace={"r", "R"}
        ) == ["[1] fiRst", "[2] second", "[3] thiRd"]
    ```
    """
    return [format_answer(answer, pattern, str_replace, idx) for idx, answer in enumerate(answers, start=1)]


def strings_to_documents(
    strings: List[str],
    meta: Union[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] = None,
    id_hash_keys: Optional[List[str]] = None,
) -> List[Document]:
    """
    Transforms a list of strings into a list of Documents. If you pass the metadata in a single
    dictionary, all Documents get the same metadata. If you pass the metadata as a list, the length of this list
    must be the same as the length of the list of strings, and each Document gets its own metadata.
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

    return [Document(content=string, meta=m, id_hash_keys=id_hash_keys) for string, m in zip(strings, all_metadata)]


def documents_to_strings(
    documents: List[Document], pattern: Optional[str] = None, str_replace: Optional[Dict[str, str]] = None
) -> List[str]:
    """
    Extracts the content field of Documents and returns a list of strings.

    Example:

    ```python
    assert documents_to_strings(
            documents=[
                Document(content="first"),
                Document(content="second"),
                Document(content="third")
            ],
            pattern="[$idx] $content",
            str_replace={"r", "R"}
        ) == ["[1] fiRst", "[2] second", "[3] thiRd"]
    ```
    """
    return [format_document(doc, pattern, str_replace, idx) for idx, doc in enumerate(documents, start=1)]


REGISTERED_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "rename": rename,
    "value_to_list": value_to_list,
    "join_lists": join_lists,
    "join_strings": join_strings,
    "join_documents": join_documents,
    "strings_to_answers": strings_to_answers,
    "answers_to_strings": answers_to_strings,
    "strings_to_documents": strings_to_documents,
    "documents_to_strings": documents_to_strings,
}


class Shaper(BaseComponent):

    """
    Shaper is a component that can invoke arbitrary, registered functions on the invocation context
    (query, documents, and so on) of a pipeline. It then passes the new or modified variables further down the pipeline.

    Using YAML configuration, the Shaper component is initialized with functions to invoke on pipeline invocation
    context.

    For example, in the YAML snippet below:
    ```yaml
        components:
        - name: shaper
          type: Shaper
          params:
            func: value_to_list
            inputs:
                value: query
                target_list: documents
            output: [questions]
    ```
    Shaper component is initialized with a directive to invoke function expand on the variable query and to store
    the result in the invocation context variable questions. All other invocation context variables are passed down
    the pipeline as they are.

    Shaper is especially useful for pipelines with PromptNodes, where we need to modify the invocation
    context to match the templates of PromptNodes.

    You can use multiple Shaper components in a pipeline to modify the invocation context as needed.

    `Shaper` supports the current functions:

    - `value_to_list`
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
        inputs: Optional[Dict[str, Union[List[str], str]]] = None,
        params: Optional[Dict[str, Any]] = None,
        publish_outputs: Union[bool, List[str]] = True,
    ):
        """
        Initializes the Shaper component.

        Some examples:

        ```yaml
        - name: shaper
          type: Shaper
          params:
          func: value_to_list
          inputs:
            value: query
            target_list: documents
          outputs:
            - questions
        ```
        This node takes the content of `query` and creates a list that contains the value of `query` `len(documents)` times.
        This list is stored in the invocation context under the key `questions`.

        ```yaml
        - name: shaper
          type: Shaper
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
        `[Document("A"), Document("B"), Document("C")]`, this shaper overwrites it with `[Document("A - B - C")]`

        ```yaml
        - name: shaper
          type: Shaper
          params:
          func: join_strings
          params:
            strings: ['a', 'b', 'c']
            delimiter: ' . '
          outputs:
            - single_string

        - name: shaper
          type: Shaper
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
            For example, `value_to_list` expects the `value` and `target_list` parameters, so `inputs` might contain:
            `{'value': 'query', 'target_list': 'documents'}`. It doesn't need to contain all keyword args, see `params`.
        :param params: Maps the function's input kwargs to some fixed values. For example, `value_to_list` expects
            `value` and `target_list` parameters, so `params` might contain
            `{'value': 'A', 'target_list': [1, 1, 1, 1]}` and the node's output is `["A", "A", "A", "A"]`.
            It doesn't need to contain all keyword args, see `inputs`.
            You can use params to provide fallback values for arguments of `run` that you're not sure exist.
            So if you need `query` to exist, you can provide a fallback value in the params, which will be used only if `query`
            is not passed to this node by the pipeline.
        :param outputs: The key to store the outputs in the invocation context. The length of the outputs must match
            the number of outputs produced by the function invoked.
        :param publish_outputs: Controls whether to publish the outputs to the pipeline's output.
            Set `True` (default value) to publishes all outputs or `False` to publish None.
            E.g. if `outputs = ["documents"]` result for `publish_outputs = True` looks like
            ```python
                {
                    "invocation_context": {
                        "documents": [...]
                    },
                    "documents": [...]
                }
            ```
            For `publish_outputs = False` result looks like
            ```python
                {
                    "invocation_context": {
                        "documents": [...]
                    },
                }
            ```
            If you want to have finer-grained control, pass a list of the outputs you want to publish.
        """
        super().__init__()
        self.function = REGISTERED_FUNCTIONS[func]
        self.outputs = outputs
        self.inputs = inputs or {}
        self.params = params or {}
        if isinstance(publish_outputs, bool):
            self.publish_outputs = self.outputs if publish_outputs else []
        else:
            self.publish_outputs = publish_outputs

    def run(  # type: ignore
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
        invocation_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict, str]:
        invocation_context = invocation_context or {}
        if query and "query" not in invocation_context.keys():
            invocation_context["query"] = query

        if file_paths and "file_paths" not in invocation_context.keys():
            invocation_context["file_paths"] = file_paths

        if labels and "labels" not in invocation_context.keys():
            invocation_context["labels"] = labels

        if documents and "documents" not in invocation_context.keys():
            invocation_context["documents"] = documents

        if meta and "meta" not in invocation_context.keys():
            invocation_context["meta"] = meta

        input_values: Dict[str, Any] = {}
        for key, value in self.inputs.items():
            if isinstance(value, list):
                input_values[key] = []
                for v in value:
                    if v in invocation_context.keys() and v is not None:
                        input_values[key].append(invocation_context[v])
            else:
                if value in invocation_context.keys() and value is not None:
                    input_values[key] = invocation_context[value]

        input_values = {**self.params, **input_values}
        try:
            logger.debug(
                "Shaper is invoking this function: %s(%s)",
                self.function.__name__,
                ", ".join([f"{key}={value}" for key, value in input_values.items()]),
            )
            output_values = self.function(**input_values)
            if not isinstance(output_values, tuple):
                output_values = (output_values,)
        except TypeError as e:
            raise ValueError(
                "Shaper couldn't apply the function to your inputs and parameters. "
                "Check the above stacktrace and make sure you provided all the correct inputs, parameters, "
                "and parameter types."
            ) from e

        if len(self.outputs) < len(output_values):
            logger.warning(
                "The number of outputs from function %s is %s. However, only %s output key(s) were provided. "
                "Only %s output(s) will be stored. "
                "Provide %s output keys to store all outputs.",
                self.function.__name__,
                len(output_values),
                len(self.outputs),
                len(self.outputs),
                len(output_values),
            )

        if len(self.outputs) > len(output_values):
            logger.warning(
                "The number of outputs from function %s is %s. However, %s output key(s) were provided. "
                "Only the first %s output key(s) will be used.",
                self.function.__name__,
                len(output_values),
                len(self.outputs),
                len(output_values),
            )

        results = {}
        for output_key, output_value in zip(self.outputs, output_values):
            invocation_context[output_key] = output_value
            if output_key in self.publish_outputs:
                results[output_key] = output_value
        results["invocation_context"] = invocation_context

        return results, "output_1"

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
