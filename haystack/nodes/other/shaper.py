from functools import reduce
import inspect
import re
from string import Template
from typing import Literal, Optional, List, Dict, Any, Tuple, Union, Callable

import logging

from haystack.nodes.base import BaseComponent
from haystack.schema import Document, Answer, MultiLabel


logger = logging.getLogger(__name__)


def rename(value: Any) -> Any:
    """
    An identity function. You can use it to rename values in the invocation context without changing them.

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
    Joins the lists you pass to it into a single list.

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
    is the content of all of the original strings separated by the delimiter you specify.

    Example:

    ```python
    assert join_strings(strings=["first", "second", "third"], delimiter=" - ", str_replace={"r": "R"}) == "fiRst - second - thiRd"
    ```
    """
    str_replace = str_replace or {}
    return delimiter.join([format_string(string, str_replace) for string in strings])


def format_string(string: str, str_replace: Optional[Dict[str, str]] = None) -> str:
    """
    Replaces strings.

    Example:

    ```python
    assert format_string(string="first", str_replace={"r": "R"}) == "fiRst"
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
    Transforms a list of documents into a list containing a single document. The content of this document
    is the joined result of all original documents, separated by the delimiter you specify.
    Use regex in the `pattern` parameter to control how each document is represented.
    You can use the following placeholders:
    - $content: The content of the document.
    - $idx: The index of the document in the list.
    - $id: The ID of the document.
    - $META_FIELD: The value of the metadata field called 'META_FIELD'.

    All metadata is dropped.

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
        str_replace={"r": "R"}
    ) == [Document(content="[1] fiRst - [2] second - [3] thiRd")]
    ```
    """
    return [Document(content=join_documents_to_string(documents, delimiter, pattern, str_replace))]


def join_documents_and_scores(documents: List[Document]) -> Tuple[List[Document]]:
    """
    Transforms a list of documents with scores in their metadata into a list containing a single document.
    The resulting document contains the scores and the contents of all the original documents.
    All metadata is dropped.
    Example:
    ```python
    assert join_documents_and_scores(
        documents=[
            Document(content="first", meta={"score": 0.9}),
            Document(content="second", meta={"score": 0.7}),
            Document(content="third", meta={"score": 0.5})
        ],
        delimiter=" - "
    ) == ([Document(content="-[0.9] first\n -[0.7] second\n -[0.5] third")], )
    ```
    """
    content = "\n".join([f"-[{round(float(doc.meta['score']),2)}] {doc.content}" for doc in documents])
    return ([Document(content=content)],)


def format_document(
    document: Document,
    pattern: Optional[str] = None,
    str_replace: Optional[Dict[str, str]] = None,
    idx: Optional[int] = None,
) -> str:
    """
    Transforms a document into a single string.
    Use regex in the `pattern` parameter to control how the document is represented.
    You can use the following placeholders:
    - $content: The content of the document.
    - $idx: The index of the document in the list.
    - $id: The ID of the document.
    - $META_FIELD: The value of the metadata field called 'META_FIELD'.

    Example:

    ```python
    assert format_document(
        document=Document(content="first"),
        pattern="prefix [$idx] $content",
        str_replace={"r": "R"},
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
    meta_params = [param for param in pattern_params if param and param not in ["content", "idx", "id"]]
    content = template.substitute(
        {
            "idx": idx,
            "content": reduce(lambda content, kv: content.replace(*kv), str_replace.items(), document.content),
            "id": reduce(lambda id, kv: id.replace(*kv), str_replace.items(), document.id),
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
    Use regex in the `pattern` parameter to control how the answer is represented.
    You can use the following placeholders:
    - $answer: The answer text.
    - $idx: The index of the answer in the list.
    - $META_FIELD: The value of the metadata field called 'META_FIELD'.

    Example:

    ```python
    assert format_answer(
        answer=Answer(answer="first"),
        pattern="prefix [$idx] $answer",
        str_replace={"r": "R"},
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
    Use regex in the `pattern` parameter to control how the documents are represented.
    You can use the following placeholders:
    - $content: The content of the document.
    - $idx: The index of the document in the list.
    - $id: The ID of the document.
    - $META_FIELD: The value of the metadata field called 'META_FIELD'.

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
        str_replace={"r": "R"}
    ) == "[1] fiRst - [2] second - [3] thiRd"
    ```
    """
    content = delimiter.join(
        format_document(doc, pattern, str_replace, idx=idx) for idx, doc in enumerate(documents, start=1)
    )
    return content


def strings_to_answers(
    strings: List[str],
    prompts: Optional[List[Union[str, List[Dict[str, str]]]]] = None,
    documents: Optional[List[Document]] = None,
    pattern: Optional[str] = None,
    reference_pattern: Optional[str] = None,
    reference_mode: Literal["index", "id", "meta"] = "index",
    reference_meta_field: Optional[str] = None,
) -> List[Answer]:
    """
    Transforms a list of strings into a list of answers.
    Specify `reference_pattern` to populate the answer's `document_ids` by extracting document references from the strings.

    :param strings: The list of strings to transform.
    :param prompts: The prompts used to generate the answers.
    :param documents: The documents used to generate the answers.
    :param pattern: The regex pattern to use for parsing the answer.
        Examples:
            `[^\\n]+$` will find "this is an answer" in string "this is an argument.\nthis is an answer".
            `Answer: (.*)` will find "this is an answer" in string "this is an argument. Answer: this is an answer".
        If None, the whole string is used as the answer. If not None, the first group of the regex is used as the answer. If there is no group, the whole match is used as the answer.
    :param reference_pattern: The regex pattern to use for parsing the document references.
        Example: `\\[(\\d+)\\]` will find "1" in string "this is an answer[1]".
        If None, no parsing is done and all documents are referenced.
    :param reference_mode: The mode used to reference documents. Supported modes are:
        - index: the document references are the one-based index of the document in the list of documents.
            Example: "this is an answer[1]" will reference the first document in the list of documents.
        - id: the document references are the document IDs.
            Example: "this is an answer[123]" will reference the document with id "123".
        - meta: the document references are the value of a metadata field of the document.
            Example: "this is an answer[123]" will reference the document with the value "123" in the metadata field specified by reference_meta_field.
    :param reference_meta_field: The name of the metadata field to use for document references in reference_mode "meta".
    :return: The list of answers.

    Examples:

    Without reference parsing:
    ```python
    assert strings_to_answers(strings=["first", "second", "third"], prompt="prompt", documents=[Document(id="123", content="content")]) == [
            Answer(answer="first", type="generative", document_ids=["123"], meta={"prompt": "prompt"}),
            Answer(answer="second", type="generative", document_ids=["123"], meta={"prompt": "prompt"}),
            Answer(answer="third", type="generative", document_ids=["123"], meta={"prompt": "prompt"}),
        ]
    ```

    With reference parsing:
    ```python
    assert strings_to_answers(strings=["first[1]", "second[2]", "third[1][3]"], prompt="prompt",
            documents=[Document(id="123", content="content"), Document(id="456", content="content"), Document(id="789", content="content")],
            reference_pattern=r"\\[(\\d+)\\]",
            reference_mode="index"
        ) == [
            Answer(answer="first", type="generative", document_ids=["123"], meta={"prompt": "prompt"}),
            Answer(answer="second", type="generative", document_ids=["456"], meta={"prompt": "prompt"}),
            Answer(answer="third", type="generative", document_ids=["123", "789"], meta={"prompt": "prompt"}),
        ]
    ```
    """
    if prompts:
        if len(prompts) == 1:
            # one prompt for all strings/documents
            documents_per_string: List[Optional[List[Document]]] = [documents] * len(strings)
            prompt_per_string: List[Optional[Union[str, List[Dict[str, str]]]]] = [prompts[0]] * len(strings)
        elif len(prompts) > 1 and len(strings) % len(prompts) == 0:
            # one prompt per string/document
            if documents is not None and len(documents) != len(prompts):
                raise ValueError("The number of documents must match the number of prompts.")
            string_multiplier = len(strings) // len(prompts)
            documents_per_string = (
                [[doc] for doc in documents for _ in range(string_multiplier)] if documents else [None] * len(strings)
            )
            prompt_per_string = [prompt for prompt in prompts for _ in range(string_multiplier)]
        else:
            raise ValueError("The number of prompts must be one or a multiple of the number of strings.")
    else:
        documents_per_string = [documents] * len(strings)
        prompt_per_string = [None] * len(strings)

    answers = []
    for string, prompt, _documents in zip(strings, prompt_per_string, documents_per_string):
        answer = string_to_answer(
            string=string,
            prompt=prompt,
            documents=_documents,
            pattern=pattern,
            reference_pattern=reference_pattern,
            reference_mode=reference_mode,
            reference_meta_field=reference_meta_field,
        )
        answers.append(answer)
    return answers


def string_to_answer(
    string: str,
    prompt: Optional[Union[str, List[Dict[str, str]]]],
    documents: Optional[List[Document]],
    pattern: Optional[str] = None,
    reference_pattern: Optional[str] = None,
    reference_mode: Literal["index", "id", "meta"] = "index",
    reference_meta_field: Optional[str] = None,
) -> Answer:
    """
    Transforms a string into an answer.
    Specify `reference_pattern` to populate the answer's `document_ids` by extracting document references from the string.

    :param string: The string to transform.
    :param prompt: The prompt used to generate the answer.
    :param documents: The documents used to generate the answer.
    :param pattern: The regex pattern to use for parsing the answer.
        Examples:
            `[^\\n]+$` will find "this is an answer" in string "this is an argument.\nthis is an answer".
            `Answer: (.*)` will find "this is an answer" in string "this is an argument. Answer: this is an answer".
        If None, the whole string is used as the answer. If not None, the first group of the regex is used as the answer. If there is no group, the whole match is used as the answer.
    :param reference_pattern: The regex pattern to use for parsing the document references.
        Example: `\\[(\\d+)\\]` will find "1" in string "this is an answer[1]".
        If None, no parsing is done and all documents are referenced.
    :param reference_mode: The mode used to reference documents. Supported modes are:
        - index: the document references are the one-based index of the document in the list of documents.
            Example: "this is an answer[1]" will reference the first document in the list of documents.
        - id: the document references are the document IDs.
            Example: "this is an answer[123]" will reference the document with id "123".
        - meta: the document references are the value of a metadata field of the document.
            Example: "this is an answer[123]" will reference the document with the value "123" in the metadata field specified by reference_meta_field.
    :param reference_meta_field: The name of the metadata field to use for document references in reference_mode "meta".
    :return: The answer
    """
    if reference_mode == "index":
        candidates = {str(idx): doc.id for idx, doc in enumerate(documents, start=1)} if documents else {}
    elif reference_mode == "id":
        candidates = {doc.id: doc.id for doc in documents} if documents else {}
    elif reference_mode == "meta":
        if not reference_meta_field:
            raise ValueError("reference_meta_field must be specified when reference_mode is 'meta'")
        candidates = (
            {doc.meta[reference_meta_field]: doc.id for doc in documents if doc.meta.get(reference_meta_field)}
            if documents
            else {}
        )
    else:
        raise ValueError(f"Invalid document_id_mode: {reference_mode}")

    if pattern:
        match = re.search(pattern, string)
        if match:
            if not match.lastindex:
                # no group in pattern -> take the whole match
                string = match.group(0)
            elif match.lastindex == 1:
                # one group in pattern -> take the group
                string = match.group(1)
            else:
                # more than one group in pattern -> raise error
                raise ValueError(f"Pattern must have at most one group: {pattern}")
        else:
            string = ""
    document_ids = parse_references(string=string, reference_pattern=reference_pattern, candidates=candidates)
    answer = Answer(answer=string, type="generative", document_ids=document_ids, meta={"prompt": prompt})
    return answer


def parse_references(
    string: str, reference_pattern: Optional[str] = None, candidates: Optional[Dict[str, str]] = None
) -> Optional[List[str]]:
    """
    Parses an answer string for document references and returns the document IDs of the referenced documents.

    :param string: The string to parse.
    :param reference_pattern: The regex pattern to use for parsing the document references.
        Example: `\\[(\\d+)\\]` will find "1" in string "this is an answer[1]".
        If None, no parsing is done and all candidate document IDs are returned.
    :param candidates: A dictionary of candidates to choose from. The keys are the reference strings and the values are the document IDs.
        If None, no parsing is done and None is returned.
    :return: A list of document IDs.
    """
    if not candidates:
        return None
    if not reference_pattern:
        return list(candidates.values())

    document_idxs = re.findall(reference_pattern, string)
    return [candidates[idx] for idx in document_idxs if idx in candidates]


def answers_to_strings(
    answers: List[Answer], pattern: Optional[str] = None, str_replace: Optional[Dict[str, str]] = None
) -> List[str]:
    """
    Extracts the content field of answers and returns a list of strings.

    Example:

    ```python
    assert answers_to_strings(
            answers=[
                Answer(answer="first"),
                Answer(answer="second"),
                Answer(answer="third")
            ],
            pattern="[$idx] $answer",
            str_replace={"r": "R"}
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
    Transforms a list of strings into a list of documents. If you pass the metadata in a single
    dictionary, all documents get the same metadata. If you pass the metadata as a list, the length of this list
    must be the same as the length of the list of strings, and each document gets its own metadata.
    You can specify `id_hash_keys` only once and it gets assigned to all documents.

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
    Extracts the content field of documents and returns a list of strings. Use regext in the `pattern` parameter to control how the documents are represented.

    Example:

    ```python
    assert documents_to_strings(
            documents=[
                Document(content="first"),
                Document(content="second"),
                Document(content="third")
            ],
            pattern="[$idx] $content",
            str_replace={"r": "R"}
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
    "join_documents_and_scores": join_documents_and_scores,
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
    the Shaper component is initialized with a directive to invoke function expand on the variable query and to store
    the result in the invocation context variable questions. All other invocation context variables are passed down
    the pipeline as they are.

    You can use multiple Shaper components in a pipeline to modify the invocation context as needed.

    Currently, `Shaper` supports the following functions:

    - `rename`
    - `value_to_list`
    - `join_lists`
    - `join_strings`
    - `format_string`
    - `join_documents`
    - `join_documents_and_scores`
    - `format_document`
    - `format_answer`
    - `join_documents_to_string`
    - `strings_to_answers`
    - `string_to_answer`
    - `parse_references`
    - `answers_to_strings`
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

        if documents != None and "documents" not in invocation_context.keys():
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

        # auto fill in input values if there's an invocation context value with the same name
        function_params = inspect.signature(self.function).parameters
        for parameter in function_params.values():
            if (
                parameter.name not in input_values.keys()
                and parameter.name not in self.params.keys()
                and parameter.name in invocation_context.keys()
            ):
                input_values[parameter.name] = invocation_context[parameter.name]

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
