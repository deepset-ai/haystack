<a id="schema"></a>

# Module schema

<a id="schema.Document"></a>

## Document

```python
@dataclass
class Document()
```

<a id="schema.Document.to_dict"></a>

#### to\_dict

```python
def to_dict(field_map={}) -> Dict
```

Convert Document to dict. An optional field_map can be supplied to change the names of the keys in the

resulting dict. This way you can work with standardized Document objects in Haystack, but adjust the format that
they are serialized / stored in other places (e.g. elasticsearch)
Example:
| doc = Document(content="some text", content_type="text")
| doc.to_dict(field_map={"custom_content_field": "content"})
| >>> {"custom_content_field": "some text", content_type": "text"}

**Arguments**:

- `field_map`: Dict with keys being the custom target keys and values being the standard Document attributes

**Returns**:

dict with content of the Document

<a id="schema.Document.from_dict"></a>

#### from\_dict

```python
@classmethod
def from_dict(cls, dict, field_map={}, id_hash_keys=None)
```

Create Document from dict. An optional field_map can be supplied to adjust for custom names of the keys in the

input dict. This way you can work with standardized Document objects in Haystack, but adjust the format that
they are serialized / stored in other places (e.g. elasticsearch)
Example:
| my_dict = {"custom_content_field": "some text", content_type": "text"}
| Document.from_dict(my_dict, field_map={"custom_content_field": "content"})

**Arguments**:

- `field_map`: Dict with keys being the custom target keys and values being the standard Document attributes

**Returns**:

dict with content of the Document

<a id="schema.Span"></a>

## Span

```python
@dataclass
class Span()
```

<a id="schema.Span.end"></a>

#### end

Defining a sequence of characters (Text span) or cells (Table span) via start and end index. 

For extractive QA: Character where answer starts/ends  
For TableQA: Cell where the answer starts/ends (counted from top left to bottom right of table)

**Arguments**:

- `start`: Position where the span starts
- `end`: Position where the spand ends

<a id="schema.Answer"></a>

## Answer

```python
@dataclass
class Answer()
```

<a id="schema.Answer.meta"></a>

#### meta

The fundamental object in Haystack to represent any type of Answers (e.g. extractive QA, generative QA or TableQA).

For example, it's used within some Nodes like the Reader, but also in the REST API.

**Arguments**:

- `answer`: The answer string. If there's no possible answer (aka "no_answer" or "is_impossible) this will be an empty string.
- `type`: One of ("generative", "extractive", "other"): Whether this answer comes from an extractive model 
(i.e. we can locate an exact answer string in one of the documents) or from a generative model 
(i.e. no pointer to a specific document, no offsets ...).
- `score`: The relevance score of the Answer determined by a model (e.g. Reader or Generator).
In the range of [0,1], where 1 means extremely relevant.
- `context`: The related content that was used to create the answer (i.e. a text passage, part of a table, image ...)
- `offsets_in_document`: List of `Span` objects with start and end positions of the answer **in the
document** (as stored in the document store).
For extractive QA: Character where answer starts => `Answer.offsets_in_document[0].start 
For TableQA: Cell where the answer starts (counted from top left to bottom right of table) => `Answer.offsets_in_document[0].start
(Note that in TableQA there can be multiple cell ranges that are relevant for the answer, thus there can be multiple `Spans` here)
- `offsets_in_context`: List of `Span` objects with start and end positions of the answer **in the
context** (i.e. the surrounding text/table of a certain window size).
For extractive QA: Character where answer starts => `Answer.offsets_in_document[0].start 
For TableQA: Cell where the answer starts (counted from top left to bottom right of table) => `Answer.offsets_in_document[0].start
(Note that in TableQA there can be multiple cell ranges that are relevant for the answer, thus there can be multiple `Spans` here)
- `document_id`: ID of the document that the answer was located it (if any)
- `meta`: Dict that can be used to associate any kind of custom meta data with the answer. 
In extractive QA, this will carry the meta data of the document where the answer was found.

<a id="schema.EvaluationResult"></a>

## EvaluationResult

```python
class EvaluationResult()
```

<a id="schema.EvaluationResult.calculate_metrics"></a>

#### calculate\_metrics

```python
def calculate_metrics(simulated_top_k_reader: int = -1, simulated_top_k_retriever: int = -1, doc_relevance_col: str = "gold_id_match", eval_mode: str = "integrated") -> Dict[str, Dict[str, float]]
```

Calculates proper metrics for each node.

For document returning nodes default metrics are:
- mrr (Mean Reciprocal Rank: see https://en.wikipedia.org/wiki/Mean_reciprocal_rank)
- map (Mean Average Precision: see https://en.wikipedia.org/wiki/Evaluation_measures_%28information_retrieval%29#Mean_average_precision)
- ndcg (Normalized Discounted Cumulative Gain: see https://en.wikipedia.org/wiki/Discounted_cumulative_gain)
- precision (Precision: How many of the returned documents were relevant?)
- recall_multi_hit (Recall according to Information Retrieval definition: How many of the relevant documents were retrieved per query?)
- recall_single_hit (Recall for Question Answering: How many of the queries returned at least one relevant document?)

For answer returning nodes default metrics are:
- exact_match (How many of the queries returned the exact answer?)
- f1 (How well do the returned results overlap with any gold answer on token basis?)
- sas if a SAS model has bin provided during during pipeline.eval() (How semantically similar is the prediction to the gold answers?)

Lower top_k values for reader and retriever than the actual values during the eval run can be simulated.
E.g. top_1_f1 for reader nodes can be calculated by setting simulated_top_k_reader=1.

Results for reader nodes with applied simulated_top_k_retriever should be considered with caution
as there are situations the result can heavily differ from an actual eval run with corresponding top_k_retriever.

**Arguments**:

- `simulated_top_k_reader`: simulates top_k param of reader
- `simulated_top_k_retriever`: simulates top_k param of retriever.
remarks: there might be a discrepancy between simulated reader metrics and an actual pipeline run with retriever top_k
- `doc_relevance_col`: column in the underlying eval table that contains the relevance criteria for documents.
values can be: 'gold_id_match', 'answer_match', 'gold_id_or_answer_match'
- `eval_mode`: the input on which the node was evaluated on.
Usually nodes get evaluated on the prediction provided by its predecessor nodes in the pipeline (value='integrated').
However, as the quality of the node itself can heavily depend on the node's input and thus the predecessor's quality,
you might want to simulate a perfect predecessor in order to get an independent upper bound of the quality of your node.
For example when evaluating the reader use value='isolated' to simulate a perfect retriever in an ExtractiveQAPipeline.
Values can be 'integrated', 'isolated'.
Default value is 'integrated'.

<a id="schema.EvaluationResult.wrong_examples"></a>

#### wrong\_examples

```python
def wrong_examples(node: str, n: int = 3, simulated_top_k_reader: int = -1, simulated_top_k_retriever: int = -1, doc_relevance_col: str = "gold_id_match", document_metric: str = "recall_single_hit", answer_metric: str = "f1", eval_mode: str = "integrated") -> List[Dict]
```

Returns the worst performing queries.

Worst performing queries are calculated based on the metric
that is either a document metric or an answer metric according to the node type.

Lower top_k values for reader and retriever than the actual values during the eval run can be simulated.
See calculate_metrics() for more information.

**Arguments**:

- `simulated_top_k_reader`: simulates top_k param of reader
- `simulated_top_k_retriever`: simulates top_k param of retriever.
remarks: there might be a discrepancy between simulated reader metrics and an actual pipeline run with retriever top_k
- `doc_relevance_col`: column that contains the relevance criteria for documents.
values can be: 'gold_id_match', 'answer_match', 'gold_id_or_answer_match'
- `document_metric`: the document metric worst queries are calculated with.
values can be: 'recall_single_hit', 'recall_multi_hit', 'mrr', 'map', 'precision'
- `document_metric`: the answer metric worst queries are calculated with.
values can be: 'f1', 'exact_match' and 'sas' if the evaluation was made using a SAS model.
- `eval_mode`: the input on which the node was evaluated on.
Usually nodes get evaluated on the prediction provided by its predecessor nodes in the pipeline (value='integrated').
However, as the quality of the node itself can heavily depend on the node's input and thus the predecessor's quality,
you might want to simulate a perfect predecessor in order to get an independent upper bound of the quality of your node.
For example when evaluating the reader use value='isolated' to simulate a perfect retriever in an ExtractiveQAPipeline.
Values can be 'integrated', 'isolated'.
Default value is 'integrated'.

<a id="schema.EvaluationResult.save"></a>

#### save

```python
def save(out_dir: Union[str, Path])
```

Saves the evaluation result.

The result of each node is saved in a separate csv with file name {node_name}.csv to the out_dir folder.

**Arguments**:

- `out_dir`: Path to the target folder the csvs will be saved.

<a id="schema.EvaluationResult.load"></a>

#### load

```python
@classmethod
def load(cls, load_dir: Union[str, Path])
```

Loads the evaluation result from disk. Expects one csv file per node. See save() for further information.

**Arguments**:

- `load_dir`: The directory containing the csv files.

