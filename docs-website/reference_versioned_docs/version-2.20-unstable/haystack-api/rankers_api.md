---
title: "Rankers"
id: rankers-api
description: "Reorders a set of Documents based on their relevance to the query."
slug: "/rankers-api"
---

<a id="hugging_face_tei"></a>

## Module hugging\_face\_tei

<a id="hugging_face_tei.TruncationDirection"></a>

### TruncationDirection

Defines the direction to truncate text when input length exceeds the model's limit.

**Attributes**:

- `LEFT` - Truncate text from the left side (start of text).
- `RIGHT` - Truncate text from the right side (end of text).

<a id="hugging_face_tei.HuggingFaceTEIRanker"></a>

### HuggingFaceTEIRanker

Ranks documents based on their semantic similarity to the query.

It can be used with a Text Embeddings Inference (TEI) API endpoint:
- [Self-hosted Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)
- [Hugging Face Inference Endpoints](https://huggingface.co/inference-endpoints)

Usage example:
```python
from haystack import Document
from haystack.components.rankers import HuggingFaceTEIRanker
from haystack.utils import Secret

reranker = HuggingFaceTEIRanker(
    url="http://localhost:8080",
    top_k=5,
    timeout=30,
    token=Secret.from_token("my_api_token")
)

docs = [Document(content="The capital of France is Paris"), Document(content="The capital of Germany is Berlin")]

result = reranker.run(query="What is the capital of France?", documents=docs)

ranked_docs = result["documents"]
print(ranked_docs)
>> {'documents': [Document(id=..., content: 'the capital of France is Paris', score: 0.9979767),
>>                Document(id=..., content: 'the capital of Germany is Berlin', score: 0.13982213)]}
```

<a id="hugging_face_tei.HuggingFaceTEIRanker.__init__"></a>

#### HuggingFaceTEIRanker.\_\_init\_\_

```python
def __init__(
    *,
    url: str,
    top_k: int = 10,
    raw_scores: bool = False,
    timeout: Optional[int] = 30,
    max_retries: int = 3,
    retry_status_codes: Optional[list[int]] = None,
    token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"],
                                                  strict=False)
) -> None
```

Initializes the TEI reranker component.

**Arguments**:

- `url`: Base URL of the TEI reranking service (for example, "https://api.example.com").
- `top_k`: Maximum number of top documents to return.
- `raw_scores`: If True, include raw relevance scores in the API payload.
- `timeout`: Request timeout in seconds.
- `max_retries`: Maximum number of retry attempts for failed requests.
- `retry_status_codes`: List of HTTP status codes that will trigger a retry.
When None, HTTP 408, 418, 429 and 503 will be retried (default: None).
- `token`: The Hugging Face token to use as HTTP bearer authorization. Not always required
depending on your TEI server configuration.
Check your HF token in your [account settings](https://huggingface.co/settings/tokens).

<a id="hugging_face_tei.HuggingFaceTEIRanker.to_dict"></a>

#### HuggingFaceTEIRanker.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="hugging_face_tei.HuggingFaceTEIRanker.from_dict"></a>

#### HuggingFaceTEIRanker.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "HuggingFaceTEIRanker"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="hugging_face_tei.HuggingFaceTEIRanker.run"></a>

#### HuggingFaceTEIRanker.run

```python
@component.output_types(documents=list[Document])
def run(
    query: str,
    documents: list[Document],
    top_k: Optional[int] = None,
    truncation_direction: Optional[TruncationDirection] = None
) -> dict[str, list[Document]]
```

Reranks the provided documents by relevance to the query using the TEI API.

**Arguments**:

- `query`: The user query string to guide reranking.
- `documents`: List of `Document` objects to rerank.
- `top_k`: Optional override for the maximum number of documents to return.
- `truncation_direction`: If set, enables text truncation in the specified direction.

**Raises**:

- `requests.exceptions.RequestException`: - If the API request fails.
- `RuntimeError`: - If the API returns an error response.

**Returns**:

A dictionary with the following keys:
- `documents`: A list of reranked documents.

<a id="hugging_face_tei.HuggingFaceTEIRanker.run_async"></a>

#### HuggingFaceTEIRanker.run\_async

```python
@component.output_types(documents=list[Document])
async def run_async(
    query: str,
    documents: list[Document],
    top_k: Optional[int] = None,
    truncation_direction: Optional[TruncationDirection] = None
) -> dict[str, list[Document]]
```

Asynchronously reranks the provided documents by relevance to the query using the TEI API.

**Arguments**:

- `query`: The user query string to guide reranking.
- `documents`: List of `Document` objects to rerank.
- `top_k`: Optional override for the maximum number of documents to return.
- `truncation_direction`: If set, enables text truncation in the specified direction.

**Raises**:

- `httpx.RequestError`: - If the API request fails.
- `RuntimeError`: - If the API returns an error response.

**Returns**:

A dictionary with the following keys:
- `documents`: A list of reranked documents.

<a id="lost_in_the_middle"></a>

## Module lost\_in\_the\_middle

<a id="lost_in_the_middle.LostInTheMiddleRanker"></a>

### LostInTheMiddleRanker

A LostInTheMiddle Ranker.

Ranks documents based on the 'lost in the middle' order so that the most relevant documents are either at the
beginning or end, while the least relevant are in the middle.

LostInTheMiddleRanker assumes that some prior component in the pipeline has already ranked documents by relevance
and requires no query as input but only documents. It is typically used as the last component before building a
prompt for an LLM to prepare the input context for the LLM.

Lost in the Middle ranking lays out document contents into LLM context so that the most relevant contents are at
the beginning or end of the input context, while the least relevant is in the middle of the context. See the
paper ["Lost in the Middle: How Language Models Use Long Contexts"](https://arxiv.org/abs/2307.03172) for more
details.

Usage example:
```python
from haystack.components.rankers import LostInTheMiddleRanker
from haystack import Document

ranker = LostInTheMiddleRanker()
docs = [Document(content="Paris"), Document(content="Berlin"), Document(content="Madrid")]
result = ranker.run(documents=docs)
for doc in result["documents"]:
    print(doc.content)
```

<a id="lost_in_the_middle.LostInTheMiddleRanker.__init__"></a>

#### LostInTheMiddleRanker.\_\_init\_\_

```python
def __init__(word_count_threshold: Optional[int] = None,
             top_k: Optional[int] = None)
```

Initialize the LostInTheMiddleRanker.

If 'word_count_threshold' is specified, this ranker includes all documents up until the point where adding
another document would exceed the 'word_count_threshold'. The last document that causes the threshold to
be breached will be included in the resulting list of documents, but all subsequent documents will be
discarded.

**Arguments**:

- `word_count_threshold`: The maximum total number of words across all documents selected by the ranker.
- `top_k`: The maximum number of documents to return.

<a id="lost_in_the_middle.LostInTheMiddleRanker.run"></a>

#### LostInTheMiddleRanker.run

```python
@component.output_types(documents=list[Document])
def run(documents: list[Document],
        top_k: Optional[int] = None,
        word_count_threshold: Optional[int] = None
        ) -> dict[str, list[Document]]
```

Reranks documents based on the "lost in the middle" order.

**Arguments**:

- `documents`: List of Documents to reorder.
- `top_k`: The maximum number of documents to return.
- `word_count_threshold`: The maximum total number of words across all documents selected by the ranker.

**Raises**:

- `ValueError`: If any of the documents is not textual.

**Returns**:

A dictionary with the following keys:
- `documents`: Reranked list of Documents

<a id="meta_field"></a>

## Module meta\_field

<a id="meta_field.MetaFieldRanker"></a>

### MetaFieldRanker

Ranks Documents based on the value of their specific meta field.

The ranking can be performed in descending order or ascending order.

Usage example:

```python
from haystack import Document
from haystack.components.rankers import MetaFieldRanker

ranker = MetaFieldRanker(meta_field="rating")
docs = [
    Document(content="Paris", meta={"rating": 1.3}),
    Document(content="Berlin", meta={"rating": 0.7}),
    Document(content="Barcelona", meta={"rating": 2.1}),
]

output = ranker.run(documents=docs)
docs = output["documents"]
assert docs[0].content == "Barcelona"
```

<a id="meta_field.MetaFieldRanker.__init__"></a>

#### MetaFieldRanker.\_\_init\_\_

```python
def __init__(meta_field: str,
             weight: float = 1.0,
             top_k: Optional[int] = None,
             ranking_mode: Literal["reciprocal_rank_fusion",
                                   "linear_score"] = "reciprocal_rank_fusion",
             sort_order: Literal["ascending", "descending"] = "descending",
             missing_meta: Literal["drop", "top", "bottom"] = "bottom",
             meta_value_type: Optional[Literal["float", "int",
                                               "date"]] = None)
```

Creates an instance of MetaFieldRanker.

**Arguments**:

- `meta_field`: The name of the meta field to rank by.
- `weight`: In range [0,1].
0 disables ranking by a meta field.
0.5 ranking from previous component and based on meta field have the same weight.
1 ranking by a meta field only.
- `top_k`: The maximum number of Documents to return per query.
If not provided, the Ranker returns all documents it receives in the new ranking order.
- `ranking_mode`: The mode used to combine the Retriever's and Ranker's scores.
Possible values are 'reciprocal_rank_fusion' (default) and 'linear_score'.
Use the 'linear_score' mode only with Retrievers or Rankers that return a score in range [0,1].
- `sort_order`: Whether to sort the meta field by ascending or descending order.
Possible values are `descending` (default) and `ascending`.
- `missing_meta`: What to do with documents that are missing the sorting metadata field.
Possible values are:
- 'drop' will drop the documents entirely.
- 'top' will place the documents at the top of the metadata-sorted list
    (regardless of 'ascending' or 'descending').
- 'bottom' will place the documents at the bottom of metadata-sorted list
    (regardless of 'ascending' or 'descending').
- `meta_value_type`: Parse the meta value into the data type specified before sorting.
This will only work if all meta values stored under `meta_field` in the provided documents are strings.
For example, if we specified `meta_value_type="date"` then for the meta value `"date": "2015-02-01"`
we would parse the string into a datetime object and then sort the documents by date.
The available options are:
- 'float' will parse the meta values into floats.
- 'int' will parse the meta values into integers.
- 'date' will parse the meta values into datetime objects.
- 'None' (default) will do no parsing.

<a id="meta_field.MetaFieldRanker.run"></a>

#### MetaFieldRanker.run

```python
@component.output_types(documents=list[Document])
def run(documents: list[Document],
        top_k: Optional[int] = None,
        weight: Optional[float] = None,
        ranking_mode: Optional[Literal["reciprocal_rank_fusion",
                                       "linear_score"]] = None,
        sort_order: Optional[Literal["ascending", "descending"]] = None,
        missing_meta: Optional[Literal["drop", "top", "bottom"]] = None,
        meta_value_type: Optional[Literal["float", "int", "date"]] = None)
```

Ranks a list of Documents based on the selected meta field by:

1. Sorting the Documents by the meta field in descending or ascending order.
2. Merging the rankings from the previous component and based on the meta field according to ranking mode and
weight.
3. Returning the top-k documents.

**Arguments**:

- `documents`: Documents to be ranked.
- `top_k`: The maximum number of Documents to return per query.
If not provided, the top_k provided at initialization time is used.
- `weight`: In range [0,1].
0 disables ranking by a meta field.
0.5 ranking from previous component and based on meta field have the same weight.
1 ranking by a meta field only.
If not provided, the weight provided at initialization time is used.
- `ranking_mode`: (optional) The mode used to combine the Retriever's and Ranker's scores.
Possible values are 'reciprocal_rank_fusion' (default) and 'linear_score'.
Use the 'score' mode only with Retrievers or Rankers that return a score in range [0,1].
If not provided, the ranking_mode provided at initialization time is used.
- `sort_order`: Whether to sort the meta field by ascending or descending order.
Possible values are `descending` (default) and `ascending`.
If not provided, the sort_order provided at initialization time is used.
- `missing_meta`: What to do with documents that are missing the sorting metadata field.
Possible values are:
- 'drop' will drop the documents entirely.
- 'top' will place the documents at the top of the metadata-sorted list
    (regardless of 'ascending' or 'descending').
- 'bottom' will place the documents at the bottom of metadata-sorted list
    (regardless of 'ascending' or 'descending').
If not provided, the missing_meta provided at initialization time is used.
- `meta_value_type`: Parse the meta value into the data type specified before sorting.
This will only work if all meta values stored under `meta_field` in the provided documents are strings.
For example, if we specified `meta_value_type="date"` then for the meta value `"date": "2015-02-01"`
we would parse the string into a datetime object and then sort the documents by date.
The available options are:
-'float' will parse the meta values into floats.
-'int' will parse the meta values into integers.
-'date' will parse the meta values into datetime objects.
-'None' (default) will do no parsing.

**Raises**:

- `ValueError`: If `top_k` is not > 0.
If `weight` is not in range [0,1].
If `ranking_mode` is not 'reciprocal_rank_fusion' or 'linear_score'.
If `sort_order` is not 'ascending' or 'descending'.
If `meta_value_type` is not 'float', 'int', 'date' or `None`.

**Returns**:

A dictionary with the following keys:
- `documents`: List of Documents sorted by the specified meta field.

<a id="meta_field_grouping_ranker"></a>

## Module meta\_field\_grouping\_ranker

<a id="meta_field_grouping_ranker.MetaFieldGroupingRanker"></a>

### MetaFieldGroupingRanker

Reorders the documents by grouping them based on metadata keys.

The MetaFieldGroupingRanker can group documents by a primary metadata key `group_by`, and subgroup them with an optional
secondary key, `subgroup_by`.
Within each group or subgroup, it can also sort documents by a metadata key `sort_docs_by`.

The output is a flat list of documents ordered by `group_by` and `subgroup_by` values.
Any documents without a group are placed at the end of the list.

The proper organization of documents helps improve the efficiency and performance of subsequent processing by an LLM.

### Usage example

```python
from haystack.components.rankers import MetaFieldGroupingRanker
from haystack.dataclasses import Document


docs = [
    Document(content="Javascript is a popular programming language", meta={"group": "42", "split_id": 7, "subgroup": "subB"}),
    Document(content="Python is a popular programming language",meta={"group": "42", "split_id": 4, "subgroup": "subB"}),
    Document(content="A chromosome is a package of DNA", meta={"group": "314", "split_id": 2, "subgroup": "subC"}),
    Document(content="An octopus has three hearts", meta={"group": "11", "split_id": 2, "subgroup": "subD"}),
    Document(content="Java is a popular programming language", meta={"group": "42", "split_id": 3, "subgroup": "subB"})
]

ranker = MetaFieldGroupingRanker(group_by="group",subgroup_by="subgroup", sort_docs_by="split_id")
result = ranker.run(documents=docs)
print(result["documents"])

# [
#     Document(id=d665bbc83e52c08c3d8275bccf4f22bf2bfee21c6e77d78794627637355b8ebc,
#             content: 'Java is a popular programming language', meta: {'group': '42', 'split_id': 3, 'subgroup': 'subB'}),
#     Document(id=a20b326f07382b3cbf2ce156092f7c93e8788df5d48f2986957dce2adb5fe3c2,
#             content: 'Python is a popular programming language', meta: {'group': '42', 'split_id': 4, 'subgroup': 'subB'}),
#     Document(id=ce12919795d22f6ca214d0f161cf870993889dcb146f3bb1b3e1ffdc95be960f,
#             content: 'Javascript is a popular programming language', meta: {'group': '42', 'split_id': 7, 'subgroup': 'subB'}),
#     Document(id=d9fc857046c904e5cf790b3969b971b1bbdb1b3037d50a20728fdbf82991aa94,
#             content: 'A chromosome is a package of DNA', meta: {'group': '314', 'split_id': 2, 'subgroup': 'subC'}),
#     Document(id=6d3b7bdc13d09aa01216471eb5fb0bfdc53c5f2f3e98ad125ff6b85d3106c9a3,
#             content: 'An octopus has three hearts', meta: {'group': '11', 'split_id': 2, 'subgroup': 'subD'})
# ]
```

<a id="meta_field_grouping_ranker.MetaFieldGroupingRanker.__init__"></a>

#### MetaFieldGroupingRanker.\_\_init\_\_

```python
def __init__(group_by: str,
             subgroup_by: Optional[str] = None,
             sort_docs_by: Optional[str] = None)
```

Creates an instance of MetaFieldGroupingRanker.

**Arguments**:

- `group_by`: The metadata key to aggregate the documents by.
- `subgroup_by`: The metadata key to aggregate the documents within a group that was created by the
`group_by` key.
- `sort_docs_by`: Determines which metadata key is used to sort the documents. If not provided, the
documents within the groups or subgroups are not sorted and are kept in the same order as
they were inserted in the subgroups.

<a id="meta_field_grouping_ranker.MetaFieldGroupingRanker.run"></a>

#### MetaFieldGroupingRanker.run

```python
@component.output_types(documents=list[Document])
def run(documents: list[Document]) -> dict[str, Any]
```

Groups the provided list of documents based on the `group_by` parameter and optionally the `subgroup_by`.

The output is a list of documents reordered based on how they were grouped.

**Arguments**:

- `documents`: The list of documents to group.

**Returns**:

A dictionary with the following keys:
- documents: The list of documents ordered by the `group_by` and `subgroup_by` metadata values.

<a id="sentence_transformers_diversity"></a>

## Module sentence\_transformers\_diversity

<a id="sentence_transformers_diversity.DiversityRankingStrategy"></a>

### DiversityRankingStrategy

The strategy to use for diversity ranking.

<a id="sentence_transformers_diversity.DiversityRankingStrategy.__str__"></a>

#### DiversityRankingStrategy.\_\_str\_\_

```python
def __str__() -> str
```

Convert a Strategy enum to a string.

<a id="sentence_transformers_diversity.DiversityRankingStrategy.from_str"></a>

#### DiversityRankingStrategy.from\_str

```python
@staticmethod
def from_str(string: str) -> "DiversityRankingStrategy"
```

Convert a string to a Strategy enum.

<a id="sentence_transformers_diversity.DiversityRankingSimilarity"></a>

### DiversityRankingSimilarity

The similarity metric to use for comparing embeddings.

<a id="sentence_transformers_diversity.DiversityRankingSimilarity.__str__"></a>

#### DiversityRankingSimilarity.\_\_str\_\_

```python
def __str__() -> str
```

Convert a Similarity enum to a string.

<a id="sentence_transformers_diversity.DiversityRankingSimilarity.from_str"></a>

#### DiversityRankingSimilarity.from\_str

```python
@staticmethod
def from_str(string: str) -> "DiversityRankingSimilarity"
```

Convert a string to a Similarity enum.

<a id="sentence_transformers_diversity.SentenceTransformersDiversityRanker"></a>

### SentenceTransformersDiversityRanker

A Diversity Ranker based on Sentence Transformers.

Applies a document ranking algorithm based on one of the two strategies:

1. Greedy Diversity Order:

    Implements a document ranking algorithm that orders documents in a way that maximizes the overall diversity
    of the documents based on their similarity to the query.

    It uses a pre-trained Sentence Transformers model to embed the query and
    the documents.

2. Maximum Margin Relevance:

    Implements a document ranking algorithm that orders documents based on their Maximum Margin Relevance (MMR)
    scores.

    MMR scores are calculated for each document based on their relevance to the query and diversity from already
    selected documents. The algorithm iteratively selects documents based on their MMR scores, balancing between
    relevance to the query and diversity from already selected documents. The 'lambda_threshold' controls the
    trade-off between relevance and diversity.

### Usage example
```python
from haystack import Document
from haystack.components.rankers import SentenceTransformersDiversityRanker

ranker = SentenceTransformersDiversityRanker(model="sentence-transformers/all-MiniLM-L6-v2", similarity="cosine", strategy="greedy_diversity_order")
ranker.warm_up()

docs = [Document(content="Paris"), Document(content="Berlin")]
query = "What is the capital of germany?"
output = ranker.run(query=query, documents=docs)
docs = output["documents"]
```

<a id="sentence_transformers_diversity.SentenceTransformersDiversityRanker.__init__"></a>

#### SentenceTransformersDiversityRanker.\_\_init\_\_

```python
def __init__(
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 10,
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var(
            ["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        similarity: Union[str, DiversityRankingSimilarity] = "cosine",
        query_prefix: str = "",
        query_suffix: str = "",
        document_prefix: str = "",
        document_suffix: str = "",
        meta_fields_to_embed: Optional[list[str]] = None,
        embedding_separator: str = "\n",
        strategy: Union[str,
                        DiversityRankingStrategy] = "greedy_diversity_order",
        lambda_threshold: float = 0.5,
        model_kwargs: Optional[dict[str, Any]] = None,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
        config_kwargs: Optional[dict[str, Any]] = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch")
```

Initialize a SentenceTransformersDiversityRanker.

**Arguments**:

- `model`: Local path or name of the model in Hugging Face's model hub,
such as `'sentence-transformers/all-MiniLM-L6-v2'`.
- `top_k`: The maximum number of Documents to return per query.
- `device`: The device on which the model is loaded. If `None`, the default device is automatically
selected.
- `token`: The API token used to download private models from Hugging Face.
- `similarity`: Similarity metric for comparing embeddings. Can be set to "dot_product" (default) or
"cosine".
- `query_prefix`: A string to add to the beginning of the query text before ranking.
Can be used to prepend the text with an instruction, as required by some embedding models,
such as E5 and BGE.
- `query_suffix`: A string to add to the end of the query text before ranking.
- `document_prefix`: A string to add to the beginning of each Document text before ranking.
Can be used to prepend the text with an instruction, as required by some embedding models,
such as E5 and BGE.
- `document_suffix`: A string to add to the end of each Document text before ranking.
- `meta_fields_to_embed`: List of meta fields that should be embedded along with the Document content.
- `embedding_separator`: Separator used to concatenate the meta fields to the Document content.
- `strategy`: The strategy to use for diversity ranking. Can be either "greedy_diversity_order" or
"maximum_margin_relevance".
- `lambda_threshold`: The trade-off parameter between relevance and diversity. Only used when strategy is
"maximum_margin_relevance".
- `model_kwargs`: Additional keyword arguments for `AutoModelForSequenceClassification.from_pretrained`
when loading the model. Refer to specific model documentation for available kwargs.
- `tokenizer_kwargs`: Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
Refer to specific model documentation for available kwargs.
- `config_kwargs`: Additional keyword arguments for `AutoConfig.from_pretrained` when loading the model configuration.
- `backend`: The backend to use for the Sentence Transformers model. Choose from "torch", "onnx", or "openvino".
Refer to the [Sentence Transformers documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
for more information on acceleration and quantization options.

<a id="sentence_transformers_diversity.SentenceTransformersDiversityRanker.warm_up"></a>

#### SentenceTransformersDiversityRanker.warm\_up

```python
def warm_up()
```

Initializes the component.

<a id="sentence_transformers_diversity.SentenceTransformersDiversityRanker.to_dict"></a>

#### SentenceTransformersDiversityRanker.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="sentence_transformers_diversity.SentenceTransformersDiversityRanker.from_dict"></a>

#### SentenceTransformersDiversityRanker.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str,
                              Any]) -> "SentenceTransformersDiversityRanker"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="sentence_transformers_diversity.SentenceTransformersDiversityRanker.run"></a>

#### SentenceTransformersDiversityRanker.run

```python
@component.output_types(documents=list[Document])
def run(query: str,
        documents: list[Document],
        top_k: Optional[int] = None,
        lambda_threshold: Optional[float] = None) -> dict[str, list[Document]]
```

Rank the documents based on their diversity.

**Arguments**:

- `query`: The search query.
- `documents`: List of Document objects to be ranker.
- `top_k`: Optional. An integer to override the top_k set during initialization.
- `lambda_threshold`: Override the trade-off parameter between relevance and diversity. Only used when
strategy is "maximum_margin_relevance".

**Raises**:

- `ValueError`: If the top_k value is less than or equal to 0.
- `RuntimeError`: If the component has not been warmed up.

**Returns**:

A dictionary with the following key:
- `documents`: List of Document objects that have been selected based on the diversity ranking.

<a id="sentence_transformers_similarity"></a>

## Module sentence\_transformers\_similarity

<a id="sentence_transformers_similarity.SentenceTransformersSimilarityRanker"></a>

### SentenceTransformersSimilarityRanker

Ranks documents based on their semantic similarity to the query.

It uses a pre-trained cross-encoder model from Hugging Face to embed the query and the documents.

### Usage example

```python
from haystack import Document
from haystack.components.rankers import SentenceTransformersSimilarityRanker

ranker = SentenceTransformersSimilarityRanker()
docs = [Document(content="Paris"), Document(content="Berlin")]
query = "City in Germany"
ranker.warm_up()
result = ranker.run(query=query, documents=docs)
docs = result["documents"]
print(docs[0].content)
```

<a id="sentence_transformers_similarity.SentenceTransformersSimilarityRanker.__init__"></a>

#### SentenceTransformersSimilarityRanker.\_\_init\_\_

```python
def __init__(*,
             model: Union[str, Path] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
             device: Optional[ComponentDevice] = None,
             token: Optional[Secret] = Secret.from_env_var(
                 ["HF_API_TOKEN", "HF_TOKEN"], strict=False),
             top_k: int = 10,
             query_prefix: str = "",
             document_prefix: str = "",
             meta_fields_to_embed: Optional[list[str]] = None,
             embedding_separator: str = "\n",
             scale_score: bool = True,
             score_threshold: Optional[float] = None,
             trust_remote_code: bool = False,
             model_kwargs: Optional[dict[str, Any]] = None,
             tokenizer_kwargs: Optional[dict[str, Any]] = None,
             config_kwargs: Optional[dict[str, Any]] = None,
             backend: Literal["torch", "onnx", "openvino"] = "torch",
             batch_size: int = 16)
```

Creates an instance of SentenceTransformersSimilarityRanker.

**Arguments**:

- `model`: The ranking model. Pass a local path or the Hugging Face model name of a cross-encoder model.
- `device`: The device on which the model is loaded. If `None`, the default device is automatically selected.
- `token`: The API token to download private models from Hugging Face.
- `top_k`: The maximum number of documents to return per query.
- `query_prefix`: A string to add at the beginning of the query text before ranking.
Use it to prepend the text with an instruction, as required by reranking models like `bge`.
- `document_prefix`: A string to add at the beginning of each document before ranking. You can use it to prepend the document
with an instruction, as required by embedding models like `bge`.
- `meta_fields_to_embed`: List of metadata fields to embed with the document.
- `embedding_separator`: Separator to concatenate metadata fields to the document.
- `scale_score`: If `True`, scales the raw logit predictions using a Sigmoid activation function.
If `False`, disables scaling of the raw logit predictions.
- `score_threshold`: Use it to return documents with a score above this threshold only.
- `trust_remote_code`: If `False`, allows only Hugging Face verified model architectures.
If `True`, allows custom models and scripts.
- `model_kwargs`: Additional keyword arguments for `AutoModelForSequenceClassification.from_pretrained`
when loading the model. Refer to specific model documentation for available kwargs.
- `tokenizer_kwargs`: Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
Refer to specific model documentation for available kwargs.
- `config_kwargs`: Additional keyword arguments for `AutoConfig.from_pretrained` when loading the model configuration.
- `backend`: The backend to use for the Sentence Transformers model. Choose from "torch", "onnx", or "openvino".
Refer to the [Sentence Transformers documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
for more information on acceleration and quantization options.
- `batch_size`: The batch size to use for inference. The higher the batch size, the more memory is required.
If you run into memory issues, reduce the batch size.

**Raises**:

- `ValueError`: If `top_k` is not > 0.

<a id="sentence_transformers_similarity.SentenceTransformersSimilarityRanker.warm_up"></a>

#### SentenceTransformersSimilarityRanker.warm\_up

```python
def warm_up() -> None
```

Initializes the component.

<a id="sentence_transformers_similarity.SentenceTransformersSimilarityRanker.to_dict"></a>

#### SentenceTransformersSimilarityRanker.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="sentence_transformers_similarity.SentenceTransformersSimilarityRanker.from_dict"></a>

#### SentenceTransformersSimilarityRanker.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str,
                              Any]) -> "SentenceTransformersSimilarityRanker"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="sentence_transformers_similarity.SentenceTransformersSimilarityRanker.run"></a>

#### SentenceTransformersSimilarityRanker.run

```python
@component.output_types(documents=list[Document])
def run(*,
        query: str,
        documents: list[Document],
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        score_threshold: Optional[float] = None) -> dict[str, list[Document]]
```

Returns a list of documents ranked by their similarity to the given query.

**Arguments**:

- `query`: The input query to compare the documents to.
- `documents`: A list of documents to be ranked.
- `top_k`: The maximum number of documents to return.
- `scale_score`: If `True`, scales the raw logit predictions using a Sigmoid activation function.
If `False`, disables scaling of the raw logit predictions.
If set, overrides the value set at initialization.
- `score_threshold`: Use it to return documents only with a score above this threshold.
If set, overrides the value set at initialization.

**Raises**:

- `ValueError`: If `top_k` is not > 0.
- `RuntimeError`: If the model is not loaded because `warm_up()` was not called before.

**Returns**:

A dictionary with the following keys:
- `documents`: A list of documents closest to the query, sorted from most similar to least similar.

<a id="transformers_similarity"></a>

## Module transformers\_similarity

<a id="transformers_similarity.TransformersSimilarityRanker"></a>

### TransformersSimilarityRanker

Ranks documents based on their semantic similarity to the query.

It uses a pre-trained cross-encoder model from Hugging Face to embed the query and the documents.

**Notes**:

  This component is considered legacy and will no longer receive updates. It may be deprecated in a future release,
  with removal following after a deprecation period.
  Consider using SentenceTransformersSimilarityRanker instead, which provides the same functionality along with
  additional features.

  ### Usage example

```python
from haystack import Document
from haystack.components.rankers import TransformersSimilarityRanker

ranker = TransformersSimilarityRanker()
docs = [Document(content="Paris"), Document(content="Berlin")]
query = "City in Germany"
ranker.warm_up()
result = ranker.run(query=query, documents=docs)
docs = result["documents"]
print(docs[0].content)
```

<a id="transformers_similarity.TransformersSimilarityRanker.__init__"></a>

#### TransformersSimilarityRanker.\_\_init\_\_

```python
def __init__(model: Union[str, Path] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
             device: Optional[ComponentDevice] = None,
             token: Optional[Secret] = Secret.from_env_var(
                 ["HF_API_TOKEN", "HF_TOKEN"], strict=False),
             top_k: int = 10,
             query_prefix: str = "",
             document_prefix: str = "",
             meta_fields_to_embed: Optional[list[str]] = None,
             embedding_separator: str = "\n",
             scale_score: bool = True,
             calibration_factor: Optional[float] = 1.0,
             score_threshold: Optional[float] = None,
             model_kwargs: Optional[dict[str, Any]] = None,
             tokenizer_kwargs: Optional[dict[str, Any]] = None,
             batch_size: int = 16)
```

Creates an instance of TransformersSimilarityRanker.

**Arguments**:

- `model`: The ranking model. Pass a local path or the Hugging Face model name of a cross-encoder model.
- `device`: The device on which the model is loaded. If `None`, overrides the default device.
- `token`: The API token to download private models from Hugging Face.
- `top_k`: The maximum number of documents to return per query.
- `query_prefix`: A string to add at the beginning of the query text before ranking.
Use it to prepend the text with an instruction, as required by reranking models like `bge`.
- `document_prefix`: A string to add at the beginning of each document before ranking. You can use it to prepend the document
with an instruction, as required by embedding models like `bge`.
- `meta_fields_to_embed`: List of metadata fields to embed with the document.
- `embedding_separator`: Separator to concatenate metadata fields to the document.
- `scale_score`: If `True`, scales the raw logit predictions using a Sigmoid activation function.
If `False`, disables scaling of the raw logit predictions.
- `calibration_factor`: Use this factor to calibrate probabilities with `sigmoid(logits * calibration_factor)`.
Used only if `scale_score` is `True`.
- `score_threshold`: Use it to return documents with a score above this threshold only.
- `model_kwargs`: Additional keyword arguments for `AutoModelForSequenceClassification.from_pretrained`
when loading the model. Refer to specific model documentation for available kwargs.
- `tokenizer_kwargs`: Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
Refer to specific model documentation for available kwargs.
- `batch_size`: The batch size to use for inference. The higher the batch size, the more memory is required.
If you run into memory issues, reduce the batch size.

**Raises**:

- `ValueError`: If `top_k` is not > 0.
If `scale_score` is True and `calibration_factor` is not provided.

<a id="transformers_similarity.TransformersSimilarityRanker.warm_up"></a>

#### TransformersSimilarityRanker.warm\_up

```python
def warm_up()
```

Initializes the component.

<a id="transformers_similarity.TransformersSimilarityRanker.to_dict"></a>

#### TransformersSimilarityRanker.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="transformers_similarity.TransformersSimilarityRanker.from_dict"></a>

#### TransformersSimilarityRanker.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "TransformersSimilarityRanker"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="transformers_similarity.TransformersSimilarityRanker.run"></a>

#### TransformersSimilarityRanker.run

```python
@component.output_types(documents=list[Document])
def run(query: str,
        documents: list[Document],
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        calibration_factor: Optional[float] = None,
        score_threshold: Optional[float] = None)
```

Returns a list of documents ranked by their similarity to the given query.

**Arguments**:

- `query`: The input query to compare the documents to.
- `documents`: A list of documents to be ranked.
- `top_k`: The maximum number of documents to return.
- `scale_score`: If `True`, scales the raw logit predictions using a Sigmoid activation function.
If `False`, disables scaling of the raw logit predictions.
- `calibration_factor`: Use this factor to calibrate probabilities with `sigmoid(logits * calibration_factor)`.
Used only if `scale_score` is `True`.
- `score_threshold`: Use it to return documents only with a score above this threshold.

**Raises**:

- `ValueError`: If `top_k` is not > 0.
If `scale_score` is True and `calibration_factor` is not provided.
- `RuntimeError`: If the model is not loaded because `warm_up()` was not called before.

**Returns**:

A dictionary with the following keys:
- `documents`: A list of documents closest to the query, sorted from most similar to least similar.
