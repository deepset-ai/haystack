<a id="base"></a>

# Module base

<a id="base.BaseGraphRetriever"></a>

## BaseGraphRetriever

```python
class BaseGraphRetriever(BaseComponent)
```

Base classfor knowledge graph retrievers.

<a id="base.BaseRetriever"></a>

## BaseRetriever

```python
class BaseRetriever(BaseComponent)
```

Base class for regular retrievers.

<a id="base.BaseRetriever.retrieve"></a>

#### retrieve

```python
@abstractmethod
def retrieve(query: str, filters: dict = None, top_k: Optional[int] = None, index: str = None, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the query.

**Arguments**:

- `query`: The query
- `filters`: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `headers`: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)

<a id="base.BaseRetriever.timing"></a>

#### timing

```python
def timing(fn, attr_name)
```

Wrapper method used to time functions.

<a id="base.BaseRetriever.eval"></a>

#### eval

```python
def eval(label_index: str = "label", doc_index: str = "eval_document", label_origin: str = "gold-label", top_k: int = 10, open_domain: bool = False, return_preds: bool = False, headers: Optional[Dict[str, str]] = None) -> dict
```

Performs evaluation on the Retriever.

Retriever is evaluated based on whether it finds the correct document given the query string and at which
position in the ranking of documents the correct document is.

|  Returns a dict containing the following metrics:

    - "recall": Proportion of questions for which correct document is among retrieved documents
    - "mrr": Mean of reciprocal rank. Rewards retrievers that give relevant documents a higher rank.
      Only considers the highest ranked relevant document.
    - "map": Mean of average precision for each question. Rewards retrievers that give relevant
      documents a higher rank. Considers all retrieved relevant documents. If ``open_domain=True``,
      average precision is normalized by the number of retrieved relevant documents per query.
      If ``open_domain=False``, average precision is normalized by the number of all relevant documents
      per query.

**Arguments**:

- `label_index`: Index/Table in DocumentStore where labeled questions are stored
- `doc_index`: Index/Table in DocumentStore where documents that are used for evaluation are stored
- `top_k`: How many documents to return per query
- `open_domain`: If ``True``, retrieval will be evaluated by checking if the answer string to a question is
contained in the retrieved docs (common approach in open-domain QA).
If ``False``, retrieval uses a stricter evaluation that checks if the retrieved document ids
are within ids explicitly stated in the labels.
- `return_preds`: Whether to add predictions in the returned dictionary. If True, the returned dictionary
contains the keys "predictions" and "metrics".
- `headers`: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)

<a id="sparse"></a>

# Module sparse

<a id="sparse.ElasticsearchRetriever"></a>

## ElasticsearchRetriever

```python
class ElasticsearchRetriever(BaseRetriever)
```

<a id="sparse.ElasticsearchRetriever.__init__"></a>

#### \_\_init\_\_

```python
def __init__(document_store: KeywordDocumentStore, top_k: int = 10, custom_query: str = None)
```

**Arguments**:

- `document_store`: an instance of an ElasticsearchDocumentStore to retrieve documents from.
- `custom_query`: query string as per Elasticsearch DSL with a mandatory query placeholder(query).
 Optionally, ES `filter` clause can be added where the values of `terms` are placeholders
 that get substituted during runtime. The placeholder(${filter_name_1}, ${filter_name_2}..)
 names must match with the filters dict supplied in self.retrieve().
 ::

     **An example custom_query:**
     ```python
    |    {
    |        "size": 10,
    |        "query": {
    |            "bool": {
    |                "should": [{"multi_match": {
    |                    "query": ${query},                 // mandatory query placeholder
    |                    "type": "most_fields",
    |                    "fields": ["content", "title"]}}],
    |                "filter": [                                 // optional custom filters
    |                    {"terms": {"year": ${years}}},
    |                    {"terms": {"quarter": ${quarters}}},
    |                    {"range": {"date": {"gte": ${date}}}}
    |                    ],
    |            }
    |        },
    |    }
     ```

 **For this custom_query, a sample retrieve() could be:**
 ```python
|    self.retrieve(query="Why did the revenue increase?",
|                  filters={"years": ["2019"], "quarters": ["Q1", "Q2"]})
```

 Optionally, highlighting can be defined by specifying Elasticsearch's highlight settings.
 See https://www.elastic.co/guide/en/elasticsearch/reference/current/highlighting.html.
 You will find the highlighted output in the returned Document's meta field by key "highlighted".
 ::

     **Example custom_query with highlighting:**
     ```python
    |    {
    |        "size": 10,
    |        "query": {
    |            "bool": {
    |                "should": [{"multi_match": {
    |                    "query": ${query},                 // mandatory query placeholder
    |                    "type": "most_fields",
    |                    "fields": ["content", "title"]}}],
    |            }
    |        },
    |        "highlight": {             // enable highlighting
    |            "fields": {            // for fields content and title
    |                "content": {},
    |                "title": {}
    |            }
    |        },
    |    }
     ```

     **For this custom_query, highlighting info can be accessed by:**
    ```python
    |    docs = self.retrieve(query="Why did the revenue increase?")
    |    highlighted_content = docs[0].meta["highlighted"]["content"]
    |    highlighted_title = docs[0].meta["highlighted"]["title"]
    ```
- `top_k`: How many documents to return per query.

<a id="sparse.ElasticsearchRetriever.retrieve"></a>

#### retrieve

```python
def retrieve(query: str, filters: dict = None, top_k: Optional[int] = None, index: str = None, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the query.

**Arguments**:

- `query`: The query
- `filters`: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.

<a id="sparse.ElasticsearchFilterOnlyRetriever"></a>

## ElasticsearchFilterOnlyRetriever

```python
class ElasticsearchFilterOnlyRetriever(ElasticsearchRetriever)
```

Naive "Retriever" that returns all documents that match the given filters. No impact of query at all.
Helpful for benchmarking, testing and if you want to do QA on small documents without an "active" retriever.

<a id="sparse.ElasticsearchFilterOnlyRetriever.retrieve"></a>

#### retrieve

```python
def retrieve(query: str, filters: dict = None, top_k: Optional[int] = None, index: str = None, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the query.

**Arguments**:

- `query`: The query
- `filters`: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents
- `headers`: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.

<a id="sparse.TfidfRetriever"></a>

## TfidfRetriever

```python
class TfidfRetriever(BaseRetriever)
```

Read all documents from a SQL backend.

Split documents into smaller units (eg, paragraphs or pages) to reduce the
computations when text is passed on to a Reader for QA.

It uses sklearn's TfidfVectorizer to compute a tf-idf matrix.

<a id="sparse.TfidfRetriever.__init__"></a>

#### \_\_init\_\_

```python
def __init__(document_store: BaseDocumentStore, top_k: int = 10, auto_fit=True)
```

**Arguments**:

- `document_store`: an instance of a DocumentStore to retrieve documents from.
- `top_k`: How many documents to return per query.
- `auto_fit`: Whether to automatically update tf-idf matrix by calling fit() after new documents have been added

<a id="sparse.TfidfRetriever.retrieve"></a>

#### retrieve

```python
def retrieve(query: str, filters: dict = None, top_k: Optional[int] = None, index: str = None, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the query.

**Arguments**:

- `query`: The query
- `filters`: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents

<a id="sparse.TfidfRetriever.fit"></a>

#### fit

```python
def fit()
```

Performing training on this class according to the TF-IDF algorithm.

<a id="dense"></a>

# Module dense

<a id="dense.DensePassageRetriever"></a>

## DensePassageRetriever

```python
class DensePassageRetriever(BaseRetriever)
```

Retriever that uses a bi-encoder (one transformer for query, one transformer for passage).
See the original paper for more details:
Karpukhin, Vladimir, et al. (2020): "Dense Passage Retrieval for Open-Domain Question Answering."
(https://arxiv.org/abs/2004.04906).

<a id="dense.DensePassageRetriever.__init__"></a>

#### \_\_init\_\_

```python
def __init__(document_store: BaseDocumentStore, query_embedding_model: Union[Path, str] = "facebook/dpr-question_encoder-single-nq-base", passage_embedding_model: Union[Path, str] = "facebook/dpr-ctx_encoder-single-nq-base", model_version: Optional[str] = None, max_seq_len_query: int = 64, max_seq_len_passage: int = 256, top_k: int = 10, use_gpu: bool = True, batch_size: int = 16, embed_title: bool = True, use_fast_tokenizers: bool = True, infer_tokenizer_classes: bool = False, similarity_function: str = "dot_product", global_loss_buffer_size: int = 150000, progress_bar: bool = True, devices: Optional[List[Union[int, str, torch.device]]] = None, use_auth_token: Optional[Union[str, bool]] = None)
```

Init the Retriever incl. the two encoder models from a local or remote model checkpoint.

The checkpoint format matches huggingface transformers' model format

**Example:**

        ```python
        |    # remote model from FAIR
        |    DensePassageRetriever(document_store=your_doc_store,
        |                          query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        |                          passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base")
        |    # or from local path
        |    DensePassageRetriever(document_store=your_doc_store,
        |                          query_embedding_model="model_directory/question-encoder",
        |                          passage_embedding_model="model_directory/context-encoder")
        ```

**Arguments**:

- `document_store`: An instance of DocumentStore from which to retrieve documents.
- `query_embedding_model`: Local path or remote name of question encoder checkpoint. The format equals the
one used by hugging-face transformers' modelhub models
Currently available remote names: ``"facebook/dpr-question_encoder-single-nq-base"``
- `passage_embedding_model`: Local path or remote name of passage encoder checkpoint. The format equals the
one used by hugging-face transformers' modelhub models
Currently available remote names: ``"facebook/dpr-ctx_encoder-single-nq-base"``
- `model_version`: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
- `max_seq_len_query`: Longest length of each query sequence. Maximum number of tokens for the query text. Longer ones will be cut down."
- `max_seq_len_passage`: Longest length of each passage/context sequence. Maximum number of tokens for the passage text. Longer ones will be cut down."
- `top_k`: How many documents to return per query.
- `use_gpu`: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
- `batch_size`: Number of questions or passages to encode at once. In case of multiple gpus, this will be the total batch size.
- `embed_title`: Whether to concatenate title and passage to a text pair that is then used to create the embedding.
This is the approach used in the original paper and is likely to improve performance if your
titles contain meaningful information for retrieval (topic, entities etc.) .
The title is expected to be present in doc.meta["name"] and can be supplied in the documents
before writing them to the DocumentStore like this:
{"text": "my text", "meta": {"name": "my title"}}.
- `use_fast_tokenizers`: Whether to use fast Rust tokenizers
- `infer_tokenizer_classes`: Whether to infer tokenizer class from the model config / name.
If `False`, the class always loads `DPRQuestionEncoderTokenizer` and `DPRContextEncoderTokenizer`.
- `similarity_function`: Which function to apply for calculating the similarity of query and passage embeddings during training.
Options: `dot_product` (Default) or `cosine`
- `global_loss_buffer_size`: Buffer size for all_gather() in DDP.
Increase if errors like "encoded data exceeds max_size ..." come up
- `progress_bar`: Whether to show a tqdm progress bar or not.
Can be helpful to disable in production deployments to keep the logs clean.
- `devices`: List of GPU devices to limit inference to certain GPUs and not use all available ones (e.g. ["cuda:0"]).
As multi-GPU training is currently not implemented for DPR, training will only use the first device provided in this list.
- `use_auth_token`: API token used to download private models from Huggingface. If this parameter is set to `True`,
the local token will be used, which must be previously created via `transformer-cli login`.
Additional information can be found here https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained

<a id="dense.DensePassageRetriever.retrieve"></a>

#### retrieve

```python
def retrieve(query: str, filters: dict = None, top_k: Optional[int] = None, index: str = None, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the query.

**Arguments**:

- `query`: The query
- `filters`: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents

<a id="dense.DensePassageRetriever.embed_queries"></a>

#### embed\_queries

```python
def embed_queries(texts: List[str]) -> List[np.ndarray]
```

Create embeddings for a list of queries using the query encoder

**Arguments**:

- `texts`: Queries to embed

**Returns**:

Embeddings, one per input queries

<a id="dense.DensePassageRetriever.embed_documents"></a>

#### embed\_documents

```python
def embed_documents(docs: List[Document]) -> List[np.ndarray]
```

Create embeddings for a list of documents using the passage encoder

**Arguments**:

- `docs`: List of Document objects used to represent documents / passages in a standardized way within Haystack.

**Returns**:

Embeddings of documents / passages shape (batch_size, embedding_dim)

<a id="dense.DensePassageRetriever.train"></a>

#### train

```python
def train(data_dir: str, train_filename: str, dev_filename: str = None, test_filename: str = None, max_samples: int = None, max_processes: int = 128, multiprocessing_strategy: Optional[str] = None, dev_split: float = 0, batch_size: int = 2, embed_title: bool = True, num_hard_negatives: int = 1, num_positives: int = 1, n_epochs: int = 3, evaluate_every: int = 1000, n_gpu: int = 1, learning_rate: float = 1e-5, epsilon: float = 1e-08, weight_decay: float = 0.0, num_warmup_steps: int = 100, grad_acc_steps: int = 1, use_amp: str = None, optimizer_name: str = "AdamW", optimizer_correct_bias: bool = True, save_dir: str = "../saved_models/dpr", query_encoder_save_dir: str = "query_encoder", passage_encoder_save_dir: str = "passage_encoder")
```

train a DensePassageRetrieval model

**Arguments**:

- `data_dir`: Directory where training file, dev file and test file are present
- `train_filename`: training filename
- `dev_filename`: development set filename, file to be used by model in eval step of training
- `test_filename`: test set filename, file to be used by model in test step after training
- `max_samples`: maximum number of input samples to convert. Can be used for debugging a smaller dataset.
- `max_processes`: the maximum number of processes to spawn in the multiprocessing.Pool used in DataSilo.
It can be set to 1 to disable the use of multiprocessing or make debugging easier.
- `multiprocessing_strategy`: Set the multiprocessing sharing strategy, this can be one of file_descriptor/file_system depending on your OS.
If your system has low limits for the number of open file descriptors, and you can’t raise them,
you should use the file_system strategy.
- `dev_split`: The proportion of the train set that will sliced. Only works if dev_filename is set to None
- `batch_size`: total number of samples in 1 batch of data
- `embed_title`: whether to concatenate passage title with each passage. The default setting in official DPR embeds passage title with the corresponding passage
- `num_hard_negatives`: number of hard negative passages(passages which are very similar(high score by BM25) to query but do not contain the answer
- `num_positives`: number of positive passages
- `n_epochs`: number of epochs to train the model on
- `evaluate_every`: number of training steps after evaluation is run
- `n_gpu`: number of gpus to train on
- `learning_rate`: learning rate of optimizer
- `epsilon`: epsilon parameter of optimizer
- `weight_decay`: weight decay parameter of optimizer
- `grad_acc_steps`: number of steps to accumulate gradient over before back-propagation is done
- `use_amp`: Whether to use automatic mixed precision (AMP) or not. The options are:
"O0" (FP32)
"O1" (Mixed Precision)
"O2" (Almost FP16)
"O3" (Pure FP16).
For more information, refer to: https://nvidia.github.io/apex/amp.html
- `optimizer_name`: what optimizer to use (default: AdamW)
- `num_warmup_steps`: number of warmup steps
- `optimizer_correct_bias`: Whether to correct bias in optimizer
- `save_dir`: directory where models are saved
- `query_encoder_save_dir`: directory inside save_dir where query_encoder model files are saved
- `passage_encoder_save_dir`: directory inside save_dir where passage_encoder model files are saved

<a id="dense.DensePassageRetriever.save"></a>

#### save

```python
def save(save_dir: Union[Path, str], query_encoder_dir: str = "query_encoder", passage_encoder_dir: str = "passage_encoder")
```

Save DensePassageRetriever to the specified directory.

**Arguments**:

- `save_dir`: Directory to save to.
- `query_encoder_dir`: Directory in save_dir that contains query encoder model.
- `passage_encoder_dir`: Directory in save_dir that contains passage encoder model.

**Returns**:

None

<a id="dense.DensePassageRetriever.load"></a>

#### load

```python
@classmethod
def load(cls, load_dir: Union[Path, str], document_store: BaseDocumentStore, max_seq_len_query: int = 64, max_seq_len_passage: int = 256, use_gpu: bool = True, batch_size: int = 16, embed_title: bool = True, use_fast_tokenizers: bool = True, similarity_function: str = "dot_product", query_encoder_dir: str = "query_encoder", passage_encoder_dir: str = "passage_encoder", infer_tokenizer_classes: bool = False)
```

Load DensePassageRetriever from the specified directory.

<a id="dense.TableTextRetriever"></a>

## TableTextRetriever

```python
class TableTextRetriever(BaseRetriever)
```

Retriever that uses a tri-encoder to jointly retrieve among a database consisting of text passages and tables
(one transformer for query, one transformer for text passages, one transformer for tables).
See the original paper for more details:
Kostić, Bogdan, et al. (2021): "Multi-modal Retrieval of Tables and Texts Using Tri-encoder Models"
(https://arxiv.org/abs/2108.04049),

<a id="dense.TableTextRetriever.__init__"></a>

#### \_\_init\_\_

```python
def __init__(document_store: BaseDocumentStore, query_embedding_model: Union[Path, str] = "deepset/bert-small-mm_retrieval-question_encoder", passage_embedding_model: Union[Path, str] = "deepset/bert-small-mm_retrieval-passage_encoder", table_embedding_model: Union[Path, str] = "deepset/bert-small-mm_retrieval-table_encoder", model_version: Optional[str] = None, max_seq_len_query: int = 64, max_seq_len_passage: int = 256, max_seq_len_table: int = 256, top_k: int = 10, use_gpu: bool = True, batch_size: int = 16, embed_meta_fields: List[str] = ["name", "section_title", "caption"], use_fast_tokenizers: bool = True, infer_tokenizer_classes: bool = False, similarity_function: str = "dot_product", global_loss_buffer_size: int = 150000, progress_bar: bool = True, devices: Optional[List[Union[int, str, torch.device]]] = None, use_auth_token: Optional[Union[str, bool]] = None)
```

Init the Retriever incl. the two encoder models from a local or remote model checkpoint.

The checkpoint format matches huggingface transformers' model format

**Arguments**:

- `document_store`: An instance of DocumentStore from which to retrieve documents.
- `query_embedding_model`: Local path or remote name of question encoder checkpoint. The format equals the
one used by hugging-face transformers' modelhub models.
- `passage_embedding_model`: Local path or remote name of passage encoder checkpoint. The format equals the
one used by hugging-face transformers' modelhub models.
- `table_embedding_model`: Local path or remote name of table encoder checkpoint. The format equala the
one used by hugging-face transformers' modelhub models.
- `model_version`: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
- `max_seq_len_query`: Longest length of each query sequence. Maximum number of tokens for the query text. Longer ones will be cut down."
- `max_seq_len_passage`: Longest length of each passage/context sequence. Maximum number of tokens for the passage text. Longer ones will be cut down."
- `top_k`: How many documents to return per query.
- `use_gpu`: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
- `batch_size`: Number of questions or passages to encode at once. In case of multiple gpus, this will be the total batch size.
- `embed_meta_fields`: Concatenate the provided meta fields and text passage / table to a text pair that is
then  used to create the embedding.
This is the approach used in the original paper and is likely to improve
performance if your titles contain meaningful information for retrieval
(topic, entities etc.).
- `use_fast_tokenizers`: Whether to use fast Rust tokenizers
- `infer_tokenizer_classes`: Whether to infer tokenizer class from the model config / name.
If `False`, the class always loads `DPRQuestionEncoderTokenizer` and `DPRContextEncoderTokenizer`.
- `similarity_function`: Which function to apply for calculating the similarity of query and passage embeddings during training.
Options: `dot_product` (Default) or `cosine`
- `global_loss_buffer_size`: Buffer size for all_gather() in DDP.
Increase if errors like "encoded data exceeds max_size ..." come up
- `progress_bar`: Whether to show a tqdm progress bar or not.
Can be helpful to disable in production deployments to keep the logs clean.
- `devices`: List of GPU devices to limit inference to certain GPUs and not use all available ones (e.g. ["cuda:0"]).
As multi-GPU training is currently not implemented for DPR, training will only use the first device provided in this list.
- `use_auth_token`: API token used to download private models from Huggingface. If this parameter is set to `True`,
the local token will be used, which must be previously created via `transformer-cli login`.
Additional information can be found here https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained

<a id="dense.TableTextRetriever.embed_queries"></a>

#### embed\_queries

```python
def embed_queries(texts: List[str]) -> List[np.ndarray]
```

Create embeddings for a list of queries using the query encoder

**Arguments**:

- `texts`: Queries to embed

**Returns**:

Embeddings, one per input queries

<a id="dense.TableTextRetriever.embed_documents"></a>

#### embed\_documents

```python
def embed_documents(docs: List[Document]) -> List[np.ndarray]
```

Create embeddings for a list of text documents and / or tables using the text passage encoder and

the table encoder.

**Arguments**:

- `docs`: List of Document objects used to represent documents / passages in
a standardized way within Haystack.

**Returns**:

Embeddings of documents / passages. Shape: (batch_size, embedding_dim)

<a id="dense.TableTextRetriever.train"></a>

#### train

```python
def train(data_dir: str, train_filename: str, dev_filename: str = None, test_filename: str = None, max_samples: int = None, max_processes: int = 128, dev_split: float = 0, batch_size: int = 2, embed_meta_fields: List[str] = ["page_title", "section_title", "caption"], num_hard_negatives: int = 1, num_positives: int = 1, n_epochs: int = 3, evaluate_every: int = 1000, n_gpu: int = 1, learning_rate: float = 1e-5, epsilon: float = 1e-08, weight_decay: float = 0.0, num_warmup_steps: int = 100, grad_acc_steps: int = 1, use_amp: str = None, optimizer_name: str = "AdamW", optimizer_correct_bias: bool = True, save_dir: str = "../saved_models/mm_retrieval", query_encoder_save_dir: str = "query_encoder", passage_encoder_save_dir: str = "passage_encoder", table_encoder_save_dir: str = "table_encoder")
```

Train a TableTextRetrieval model.

**Arguments**:

- `data_dir`: Directory where training file, dev file and test file are present.
- `train_filename`: Training filename.
- `dev_filename`: Development set filename, file to be used by model in eval step of training.
- `test_filename`: Test set filename, file to be used by model in test step after training.
- `max_samples`: Maximum number of input samples to convert. Can be used for debugging a smaller dataset.
- `max_processes`: The maximum number of processes to spawn in the multiprocessing.Pool used in DataSilo.
It can be set to 1 to disable the use of multiprocessing or make debugging easier.
- `dev_split`: The proportion of the train set that will sliced. Only works if dev_filename is set to None.
- `batch_size`: Total number of samples in 1 batch of data.
- `embed_meta_fields`: Concatenate meta fields with each passage and table.
The default setting in official MMRetrieval embeds page title,
section title and caption with the corresponding table and title with
corresponding text passage.
- `num_hard_negatives`: Number of hard negative passages (passages which are
very similar (high score by BM25) to query but do not contain the answer)-
- `num_positives`: Number of positive passages.
- `n_epochs`: Number of epochs to train the model on.
- `evaluate_every`: Number of training steps after evaluation is run.
- `n_gpu`: Number of gpus to train on.
- `learning_rate`: Learning rate of optimizer.
- `epsilon`: Epsilon parameter of optimizer.
- `weight_decay`: Weight decay parameter of optimizer.
- `grad_acc_steps`: Number of steps to accumulate gradient over before back-propagation is done.
- `use_amp`: Whether to use automatic mixed precision (AMP) or not. The options are:
"O0" (FP32)
"O1" (Mixed Precision)
"O2" (Almost FP16)
"O3" (Pure FP16).
For more information, refer to: https://nvidia.github.io/apex/amp.html
- `optimizer_name`: What optimizer to use (default: TransformersAdamW).
- `num_warmup_steps`: Number of warmup steps.
- `optimizer_correct_bias`: Whether to correct bias in optimizer.
- `save_dir`: Directory where models are saved.
- `query_encoder_save_dir`: Directory inside save_dir where query_encoder model files are saved.
- `passage_encoder_save_dir`: Directory inside save_dir where passage_encoder model files are saved.
- `table_encoder_save_dir`: Directory inside save_dir where table_encoder model files are saved.

<a id="dense.TableTextRetriever.save"></a>

#### save

```python
def save(save_dir: Union[Path, str], query_encoder_dir: str = "query_encoder", passage_encoder_dir: str = "passage_encoder", table_encoder_dir: str = "table_encoder")
```

Save TableTextRetriever to the specified directory.

**Arguments**:

- `save_dir`: Directory to save to.
- `query_encoder_dir`: Directory in save_dir that contains query encoder model.
- `passage_encoder_dir`: Directory in save_dir that contains passage encoder model.
- `table_encoder_dir`: Directory in save_dir that contains table encoder model.

**Returns**:

None

<a id="dense.TableTextRetriever.load"></a>

#### load

```python
@classmethod
def load(cls, load_dir: Union[Path, str], document_store: BaseDocumentStore, max_seq_len_query: int = 64, max_seq_len_passage: int = 256, max_seq_len_table: int = 256, use_gpu: bool = True, batch_size: int = 16, embed_meta_fields: List[str] = ["name", "section_title", "caption"], use_fast_tokenizers: bool = True, similarity_function: str = "dot_product", query_encoder_dir: str = "query_encoder", passage_encoder_dir: str = "passage_encoder", table_encoder_dir: str = "table_encoder", infer_tokenizer_classes: bool = False)
```

Load TableTextRetriever from the specified directory.

<a id="dense.EmbeddingRetriever"></a>

## EmbeddingRetriever

```python
class EmbeddingRetriever(BaseRetriever)
```

<a id="dense.EmbeddingRetriever.__init__"></a>

#### \_\_init\_\_

```python
def __init__(document_store: BaseDocumentStore, embedding_model: str, model_version: Optional[str] = None, use_gpu: bool = True, batch_size: int = 32, max_seq_len: int = 512, model_format: str = "farm", pooling_strategy: str = "reduce_mean", emb_extraction_layer: int = -1, top_k: int = 10, progress_bar: bool = True, devices: Optional[List[Union[int, str, torch.device]]] = None, use_auth_token: Optional[Union[str, bool]] = None)
```

**Arguments**:

- `document_store`: An instance of DocumentStore from which to retrieve documents.
- `embedding_model`: Local path or name of model in Hugging Face's model hub such as ``'sentence-transformers/all-MiniLM-L6-v2'``
- `model_version`: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
- `use_gpu`: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
- `batch_size`: Number of documents to encode at once.
- `max_seq_len`: Longest length of each document sequence. Maximum number of tokens for the document text. Longer ones will be cut down.
- `model_format`: Name of framework that was used for saving the model. Options:
- ``'farm'``
- ``'transformers'``
- ``'sentence_transformers'``
- `pooling_strategy`: Strategy for combining the embeddings from the model (for farm / transformers models only).
Options:

- ``'cls_token'`` (sentence vector)
- ``'reduce_mean'`` (sentence vector)
- ``'reduce_max'`` (sentence vector)
- ``'per_token'`` (individual token vectors)
- `emb_extraction_layer`: Number of layer from which the embeddings shall be extracted (for farm / transformers models only).
Default: -1 (very last layer).
- `top_k`: How many documents to return per query.
- `progress_bar`: If true displays progress bar during embedding.
- `devices`: List of GPU devices to limit inference to certain GPUs and not use all available ones (e.g. ["cuda:0"]).
As multi-GPU training is currently not implemented for DPR, training will only use the first device provided in this list.
- `use_auth_token`: API token used to download private models from Huggingface. If this parameter is set to `True`,
the local token will be used, which must be previously created via `transformer-cli login`.
Additional information can be found here https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained

<a id="dense.EmbeddingRetriever.retrieve"></a>

#### retrieve

```python
def retrieve(query: str, filters: dict = None, top_k: Optional[int] = None, index: str = None, headers: Optional[Dict[str, str]] = None) -> List[Document]
```

Scan through documents in DocumentStore and return a small number documents

that are most relevant to the query.

**Arguments**:

- `query`: The query
- `filters`: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
- `top_k`: How many documents to return per query.
- `index`: The name of the index in the DocumentStore from which to retrieve documents

<a id="dense.EmbeddingRetriever.embed_queries"></a>

#### embed\_queries

```python
def embed_queries(texts: List[str]) -> List[np.ndarray]
```

Create embeddings for a list of queries.

**Arguments**:

- `texts`: Queries to embed

**Returns**:

Embeddings, one per input queries

<a id="dense.EmbeddingRetriever.embed_documents"></a>

#### embed\_documents

```python
def embed_documents(docs: List[Document]) -> List[np.ndarray]
```

Create embeddings for a list of documents.

**Arguments**:

- `docs`: List of documents to embed

**Returns**:

Embeddings, one per input document

<a id="text2sparql"></a>

# Module text2sparql

<a id="text2sparql.Text2SparqlRetriever"></a>

## Text2SparqlRetriever

```python
class Text2SparqlRetriever(BaseGraphRetriever)
```

Graph retriever that uses a pre-trained Bart model to translate natural language questions
given in text form to queries in SPARQL format.
The generated SPARQL query is executed on a knowledge graph.

<a id="text2sparql.Text2SparqlRetriever.__init__"></a>

#### \_\_init\_\_

```python
def __init__(knowledge_graph, model_name_or_path, top_k: int = 1)
```

Init the Retriever by providing a knowledge graph and a pre-trained BART model

**Arguments**:

- `knowledge_graph`: An instance of BaseKnowledgeGraph on which to execute SPARQL queries.
- `model_name_or_path`: Name of or path to a pre-trained BartForConditionalGeneration model.
- `top_k`: How many SPARQL queries to generate per text query.

<a id="text2sparql.Text2SparqlRetriever.retrieve"></a>

#### retrieve

```python
def retrieve(query: str, top_k: Optional[int] = None)
```

Translate a text query to SPARQL and execute it on the knowledge graph to retrieve a list of answers

**Arguments**:

- `query`: Text query that shall be translated to SPARQL and then executed on the knowledge graph
- `top_k`: How many SPARQL queries to generate per text query.

<a id="text2sparql.Text2SparqlRetriever.format_result"></a>

#### format\_result

```python
def format_result(result)
```

Generate formatted dictionary output with text answer and additional info

**Arguments**:

- `result`: The result of a SPARQL query as retrieved from the knowledge graph
