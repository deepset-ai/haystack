<a name="schema"></a>
# Module schema

<a name="schema.Document"></a>
## Document

```python
@dataclass
class Document()
```

<a name="schema.Document.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(content: Union[str, pd.DataFrame], content_type: Literal["text", "table", "image"] = "text", id: Optional[str] = None, score: Optional[float] = None, meta: Dict[str, Any] = None, embedding: Optional[np.ndarray] = None, id_hash_keys: Optional[List[str]] = None)
```

One of the core data classes in Haystack. It's used to represent documents / passages in a standardized way within Haystack.
Documents are stored in DocumentStores, are returned by Retrievers, are the input for Readers and are used in
many other places that manipulate or interact with document-level data.

Note: There can be multiple Documents originating from one file (e.g. PDF), if you split the text
into smaller passages. We'll have one Document per passage in this case.

Each document has a unique ID. This can be supplied by the user or generated automatically.
It's particularly helpful for handling of duplicates and referencing documents in other objects (e.g. Labels)

There's an easy option to convert from/to dicts via `from_dict()` and `to_dict`.

**Arguments**:

- `content`: Content of the document. For most cases, this will be text, but it can be a table or image.
- `content_type`: One of "image", "table" or "image". Haystack components can use this to adjust their
                     handling of Documents and check compatibility.
- `id`: Unique ID for the document. If not supplied by the user, we'll generate one automatically by
           creating a hash from the supplied text. This behaviour can be further adjusted by `id_hash_keys`.
- `score`: The relevance score of the Document determined by a model (e.g. Retriever or Re-Ranker).
              In the range of [0,1], where 1 means extremely relevant.
- `meta`: Meta fields for a document like name, url, or author in the form of a custom dict (any keys and values allowed).
- `embedding`: Vector encoding of the text
- `id_hash_keys`: Generate the document id from a custom list of strings.
                     If you want ensure you don't have duplicate documents in your DocumentStore but texts are
                     not unique, you can provide custom strings here that will be used (e.g. ["filename_xy", "text_of_doc"].

<a name="schema.Document.to_dict"></a>
#### to\_dict

```python
 | to_dict(field_map={}) -> Dict
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

<a name="schema.Document.from_dict"></a>
#### from\_dict

```python
 | @classmethod
 | from_dict(cls, dict, field_map={})
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

<a name="schema.Document.__lt__"></a>
#### \_\_lt\_\_

```python
 | __lt__(other)
```

Enable sorting of Documents by score

<a name="schema.Span"></a>
## Span

```python
@dataclass
class Span()
```

<a name="schema.Span.end"></a>
#### end

Defining a sequence of characters (Text span) or cells (Table span) via start and end index.
For extractive QA: Character where answer starts/ends
For TableQA: Cell where the answer starts/ends (counted from top left to bottom right of table)

**Arguments**:

- `start`: Position where the span starts
- `end`: Position where the spand ends

<a name="schema.Answer"></a>
## Answer

```python
@dataclass
class Answer()
```

<a name="schema.Answer.meta"></a>
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

<a name="schema.Answer.__lt__"></a>
#### \_\_lt\_\_

```python
 | __lt__(other)
```

Enable sorting of Answers by score

<a name="schema.Label"></a>
## Label

```python
@dataclass
class Label()
```

<a name="schema.Label.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(query: str, document: Document, is_correct_answer: bool, is_correct_document: bool, origin: Literal["user-feedback", "gold-label"], answer: Optional[Answer], id: Optional[str] = None, no_answer: Optional[bool] = None, pipeline_id: Optional[str] = None, created_at: Optional[str] = None, updated_at: Optional[str] = None, meta: Optional[dict] = None)
```

Object used to represent label/feedback in a standardized way within Haystack.
This includes labels from dataset like SQuAD, annotations from labeling tools,
or, user-feedback from the Haystack REST API.

**Arguments**:

- `query`: the question (or query) for finding answers.
- `document`: 
- `answer`: the answer object.
- `is_correct_answer`: whether the sample is positive or negative.
- `is_correct_document`: in case of negative sample(is_correct_answer is False), there could be two cases;
                            incorrect answer but correct document & incorrect document. This flag denotes if
                            the returned document was correct.
- `origin`: the source for the labels. It can be used to later for filtering.
- `id`: Unique ID used within the DocumentStore. If not supplied, a uuid will be generated automatically.
- `no_answer`: whether the question in unanswerable.
- `pipeline_id`: pipeline identifier (any str) that was involved for generating this label (in-case of user feedback).
- `created_at`: Timestamp of creation with format yyyy-MM-dd HH:mm:ss.
                   Generate in Python via time.strftime("%Y-%m-%d %H:%M:%S").
- `created_at`: Timestamp of update with format yyyy-MM-dd HH:mm:ss.
                   Generate in Python via time.strftime("%Y-%m-%d %H:%M:%S")
- `meta`: Meta fields like "annotator_name" in the form of a custom dict (any keys and values allowed).

<a name="schema.MultiLabel"></a>
## MultiLabel

```python
@dataclass
class MultiLabel()
```

<a name="schema.MultiLabel.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(labels: List[Label], drop_negative_labels=False, drop_no_answers=False)
```

There are often multiple `Labels` associated with a single query. For example, there can be multiple annotated
answers for one question or multiple documents contain the information you want for a query.
This class is "syntactic sugar" that simplifies the work with such a list of related Labels.
It stored the original labels in MultiLabel.labels and provides additional aggregated attributes that are
automatically created at init time. For example, MultiLabel.no_answer allows you to easily access if any of the
underlying Labels provided a text answer and therefore demonstrates that there is indeed a possible answer.

**Arguments**:

- `labels`: A list lof labels that belong to a similar query and shall be "grouped" together
- `drop_negative_labels`: Whether to drop negative labels from that group (e.g. thumbs down feedback from UI)
- `drop_no_answers`: Whether to drop labels that specify the answer is impossible

