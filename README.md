![Logo](https://raw.githubusercontent.com/deepset-ai/haystack/master/docs/_src/img/haystack_logo_blue_banner.png)

<p>
    <a href="https://github.com/deepset-ai/haystack/actions">
        <img alt="Build" src="https://github.com/deepset-ai/haystack/workflows/Build/badge.svg?branch=master">
    </a>
    <a href="http://mypy-lang.org/">
        <img alt="Checked with MyPy" src="https://camo.githubusercontent.com/34b3a249cd6502d0a521ab2f42c8830b7cfd03fa/687474703a2f2f7777772e6d7970792d6c616e672e6f72672f7374617469632f6d7970795f62616467652e737667">
    </a>
    <a href="https://haystack.deepset.ai/docs/intromd">
        <img alt="Documentation" src="https://img.shields.io/website/http/haystack.deepset.ai/docs/intromd.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/deepset-ai/haystack/releases">
        <img alt="Release" src="https://img.shields.io/github/release/deepset-ai/haystack">
    </a>
    <a href="https://github.com/deepset-ai/haystack/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/deepset-ai/haystack.svg?color=blue">
    </a>
    <a href="https://github.com/deepset-ai/haystack/commits/master">
        <img alt="Last commit" src="https://img.shields.io/github/last-commit/deepset-ai/haystack">
    </a>
    <a href="https://pepy.tech/project/farm-haystack">
        <img alt="Downloads" src="https://pepy.tech/badge/farm-haystack/month">
    </a>
</p>

Haystack is an end-to-end framework for Question Answering & Neural search that enables you to ...

... **ask questions** in natural language and find granular answers in your own documents.  
... **do semantic document search** and retrieve more relevant documents for your search queries.  
... **search at scale** through millions of documents.  
... **use off-the-shelf models or fine-tune** them to your own domain.  
... **evaluate, benchmark and continuously improve** your models via user feedback.  
... **improve chat bots** by leveraging existing knowledge bases for the long tail of queries.  
... **automate processes** by automatically applying a list of questions to new documents and using the extracted answers.  

|  |  |
|-|-|
| :ledger: [Docs](https://haystack.deepset.ai/docs/intromd) | Usage, Guides, API documentation ...|
| :computer: [Installation](https://github.com/deepset-ai/haystack/#installation) | How to install |
| :art: [Key components](https://github.com/deepset-ai/haystack/#key-components) | Overview of core concepts |
| :eyes: [Quick Tour](https://github.com/deepset-ai/haystack/#quick-tour) | Basic explanation of concepts, options and usage |
| :mortar_board: [Tutorials](https://github.com/deepset-ai/haystack/#tutorials) | Jupyter/Colab Notebooks & Scripts |
| :bar_chart: [Benchmarks](https://haystack.deepset.ai/bm/benchmarks) | Speed & Accuracy of Retriever, Readers and DocumentStores |
| :telescope: [Roadmap](https://haystack.deepset.ai/docs/latest/roadmapmd) | Public roadmap of Haystack |
| :heart: [Contributing](https://github.com/deepset-ai/haystack/#heart-contributing) | We welcome all contributions! |
| :pray: [Slack](https://haystack.deepset.ai/community/join) | Join our community on Slack |
| :bird: [Twitter](https://twitter.com/deepset_ai) | Follow us on Twitter for news and updates |
| :newspaper: [Blog](https://medium.com/deepset-ai) | Read our articles on Medium |

## Core Features

-   **Latest models**: Utilize all latest transformer based models (e.g. BERT, RoBERTa, MiniLM) for extractive QA, generative QA and document retrieval.
-   **Modular**: Multiple choices to fit your tech stack and use case. Pick your favorite database, file converter or modeling framwework.
-   **Open**: 100% compatible with HuggingFace's model hub. Tight interfaces to other frameworks (e.g. Transformers, FARM, sentence-transformers)
-   **Scalable**: Scale to millions of docs via retrievers, production-ready backends like Elasticsearch / FAISS and a fastAPI REST API
-   **End-to-End**: All tooling in one place: file conversion, cleaning, splitting, training, eval, inference, labeling ...    
-   **Developer friendly**: Easy to debug, extend and modify.
-   **Customizable**: Fine-tune models to your own domain or implement your custom DocumentStore.
-   **Continuous Learning**: Collect new training data via user feedback in production & improve your models continuously


## Installation

PyPi:

    pip install farm-haystack

Master branch (if you wanna try the latest features):

    git clone https://github.com/deepset-ai/haystack.git
    cd haystack
    pip install --editable .

To update your installation, just do a git pull. The --editable flag
will update changes immediately.

On Windows you might need:
```
pip install farm-haystack -f https://download.pytorch.org/whl/torch_stable.html
```

## Key Components

![image](https://raw.githubusercontent.com/deepset-ai/haystack/master/docs/_src/img/concepts_haystack_v2.png)

1.  **FileConverter**: Extracts pure text from files (pdf, docx, pptx, html and many more). 
2.  **PreProcessor**: Cleans and splits texts into smaller chunks.
3.  **DocumentStore**: Database storing the documents, metadata and vectors for our search. 
We recommend Elasticsearch or FAISS, but have also more light-weight options for fast prototyping (SQL or In-Memory).
4.  **Retriever**: Fast algorithms that identify candidate documents for a given query from a large collection of documents. 
    Retrievers narrow down the search space significantly and are therefore key for scalable QA.
    Haystack supports sparse methods (TF-IDF, BM25, custom Elasticsearch queries)
    and state of the art dense methods (e.g. sentence-transformers and Dense Passage Retrieval)
5.  **Reader**: Neural network (e.g. BERT or RoBERTA) that reads through texts in detail
    to find an answer. The Reader takes multiple passages of text as input and returns top-n answers. Models are trained via [FARM](https://github.com/deepset-ai/FARM) or [Transformers](https://github.com/huggingface/transformers) on SQuAD like tasks.  You can just load a pretrained model from [Hugging Face's model hub](https://huggingface.co/models) or fine-tune it on your own domain data.
6.  **Generator**: Neural network (e.g. RAG) that *generates* an answer for a given question conditioned on the retrieved documents from the retriever.
6.  **Pipeline**: Stick building blocks together to highly custom pipelines that are represented as Directed Acyclic Graphs (DAG). Think of it as "Apache Airflow for search".
7.  **REST API**: Exposes a simple API based on fastAPI for running QA search, uploading files and collecting user feedback for continuous learning.
8.  **Haystack Annotate**: Create custom QA labels to improve performance of your domain-specific models. [Hosted version](https://annotate.deepset.ai/login) or [Docker images](https://github.com/deepset-ai/haystack/tree/master/annotation_tool). 



## Usage

```python
# DB to store your docs
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="",
                                            index="document", embedding_dim=768,
                                            embedding_field="embedding")

# Index your docs
# (Options: Convert text from PDFs etc. via FileConverter; Split and clean docs with the PreProcessor)
docs = [Document(text="Arya accompanies her father Ned and her sister Sansa to King's Landing. Before their departure ...", meta={}), 
       ...]

document_store.write_documents([doc])

# Init Retriever: Fast algorithm to identify most promising candidate docs
# (Options: DPR, TF-IDF, Elasticsearch, Plain Embeddings ..)
retriever = DensePassageRetriever(document_store=document_store,                         
                                query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                )
document_store.update_embeddings(retriever)

# Init Reader: Powerful, but slower neural model 
# (Options: FARM or Transformers Framework; Extractive or generative models)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

# The Pipeline sticks together Reader + Retriever to a DAG
# There's many different pipeline types and you can easily build your own
pipeline = ExtractiveQAPipeline(reader, retriever)

# Voilá! Ask a question!
prediction = pipeline.run(query="Who is the father of Arya Stark?", top_k_retriever=10,top_k_reader=3)
print_answers(prediction, details="minimal")

[   {   'answer': 'Eddard',
        'context': """... She travels with her father, Eddard, to 
                   King's Landing when he is made Hand of the King ..."""},
    {   'answer': 'Ned',
        'context': """... girl disguised as a boy all along and is surprised 
                   to learn she is Arya, Ned Stark's daughter ..."""},
    {   'answer': 'Ned',
        'context': """... Arya accompanies her father Ned and her sister Sansa to
                   King's Landing. Before their departure ..."""}
]
``` 
## Tutorials

-   Tutorial 1 - Basic QA Pipeline: [Jupyter notebook](https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial1_Basic_QA_Pipeline.ipynb)
    or
    [Colab](https://colab.research.google.com/github/deepset-ai/haystack/blob/master/tutorials/Tutorial1_Basic_QA_Pipeline.ipynb)
-   Tutorial 2 - Fine-tuning a model on own data: [Jupyter notebook](https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial2_Finetune_a_model_on_your_data.ipynb)
    or
    [Colab](https://colab.research.google.com/github/deepset-ai/haystack/blob/master/tutorials/Tutorial2_Finetune_a_model_on_your_data.ipynb)
-   Tutorial 3 - Basic QA Pipeline without Elasticsearch: [Jupyter notebook](https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial3_Basic_QA_Pipeline_without_Elasticsearch.ipynb)
    or
    [Colab](https://colab.research.google.com/github/deepset-ai/haystack/blob/master/tutorials/Tutorial3_Basic_QA_Pipeline_without_Elasticsearch.ipynb)
-   Tutorial 4 - FAQ-style QA: [Jupyter notebook](https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial4_FAQ_style_QA.ipynb)
    or
    [Colab](https://colab.research.google.com/github/deepset-ai/haystack/blob/master/tutorials/Tutorial4_FAQ_style_QA.ipynb)
-   Tutorial 5 - Evaluation of the whole QA-Pipeline: [Jupyter noteboook](https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial5_Evaluation.ipynb)
    or
    [Colab](https://colab.research.google.com/github/deepset-ai/haystack/blob/master/tutorials/Tutorial5_Evaluation.ipynb)
-   Tutorial 6 - Better Retrievers via "Dense Passage Retrieval":
    [Jupyter noteboook](https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial6_Better_Retrieval_via_DPR.ipynb)
    or
    [Colab](https://colab.research.google.com/github/deepset-ai/haystack/blob/master/tutorials/Tutorial6_Better_Retrieval_via_DPR.ipynb)
-   Tutorial 7 - Generative QA via "Retrieval-Augmented Generation":
    [Jupyter noteboook](https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial7_RAG_Generator.ipynb)
    or
    [Colab](https://colab.research.google.com/github/deepset-ai/haystack/blob/master/tutorials/Tutorial7_RAG_Generator.ipynb)
-   Tutorial 8 - Preprocessing:
    [Jupyter noteboook](https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial8_Preprocessing.ipynb)
    or
    [Colab](https://colab.research.google.com/github/deepset-ai/haystack/blob/master/tutorials/Tutorial8_Preprocessing.ipynb)
-   Tutorial 9 - DPR Training:
    [Jupyter noteboook](https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial9_DPR_training.ipynb)
    or
    [Colab](https://colab.research.google.com/github/deepset-ai/haystack/blob/master/tutorials/Tutorial9_DPR_training.ipynb)


## Quick Tour
[File Conversion](https://github.com/deepset-ai/haystack/blob/master/README.md#1-file-conversion) | [Preprocessing](https://github.com/deepset-ai/haystack/blob/master/README.md#2-preprocessing) | [DocumentStores](https://github.com/deepset-ai/haystack/blob/master/README.md#3-documentstores) | [Retrievers](https://github.com/deepset-ai/haystack/blob/master/README.md#4-retrievers) | [Readers](https://github.com/deepset-ai/haystack/blob/master/README.md#5-readers) | [Pipelines](https://github.com/deepset-ai/haystack/blob/master/README.md#6-pipelines) | [REST API](https://github.com/deepset-ai/haystack/blob/master/README.md#7-rest-api) |  [Labeling Tool](https://github.com/deepset-ai/haystack/blob/master/README.md#8-labeling-tool) 

### 1) File Conversion
**What**  
Different converters to extract text from your original files (pdf, docx, txt, html).
While it's almost impossible to cover all types, layouts and special cases (especially in PDFs), we cover the most common formats (incl. multi-column) and extract meta information (e.g. page splits). The converters are easily extendable, so that you can customize them for your files if needed.

**Available options**  
- Txt
- PDF 
- Docx
- Apache Tika (Supports > 340 file formats)

**Example**  

```python
#PDF
from haystack.file_converter.pdf import PDFToTextConverter
converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["de","en"])
doc = converter.convert(file_path=file, meta=None)
# => {"text": "text first page \f text second page ...", "meta": None}

#DOCX
from haystack.file_converter.docx import DocxToTextConverter
converter = DocxToTextConverter(remove_numeric_tables=True, valid_languages=["de","en"])
doc = converter.convert(file_path=file, meta=None)
# => {"text": "some text", "meta": None}
```


### 2) Preprocessing
**What**  
Cleaning and splitting of your texts are crucial steps that will directly impact the speed and accuracy of your search.
The splitting of larger texts is especially important for achieving fast query speed. The longer the texts that the retriever passes to the reader, the slower your queries. 

**Available Options**  
We provide a basic `PreProcessor` class that allows: 
- clean whitespace, headers, footer and empty lines
- split by words, sentences or passages
- option for "overlapping" splits 
- option to never split within a sentence

You can easily extend this class to your own custom requirements. 

**Example**  
```python 
converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["en"])

processor = PreProcessor(clean_empty_lines=True,
                         clean_whitespace=True,
                         clean_header_footer=True,
                         split_by="word",
                         split_length=200,
                         split_respect_sentence_boundary=True)
docs = []
for f_name, f_path in zip(filenames, filepaths):
    # Optional: Supply any meta data here
    # the "name" field will be used by DPR if embed_title=True, rest is custom and can be named arbitrarily
    cur_meta = {"name": f_name, "category": "a" ...}
    
    # Run the conversion on each file (PDF -> 1x doc)
    d = converter.convert(f_path, meta=cur_meta)
    
    # clean and split each dict (1x doc -> multiple docs)
    d = processor.process(d)
    docs.extend(d)

# at this point docs will be [{"text": "some", "meta":{"name": "myfilename", "category":"a"}},...]
document_store.write_documents(docs)
```

### 3) DocumentStores

**What**  
-  Store your texts, meta data and optionally embeddings
-  Documents should be chunked into smaller units (e.g. paragraphs)
    before indexing to make the results returned by the Retriever more
    granular and accurate.
    
**Available Options**  

- Elasticsearch
- FAISS
- SQL
- InMemory

**Example**  

```python 

# Run elasticsearch, e.g. via docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.6.2

# Connect 
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

# Get all documents
document_store.get_all_documents()

# Query
document_store.query(query="What is the meaning of life?", filters=None, top_k=5)
document_store.query_by_embedding(query_emb, filters=None, top_k=5)

``` 
-> See [docs](https://haystack.deepset.ai/docs/latest/documentstoremd) for details


### 4) Retrievers

**What**  
The Retriever is a fast "filter" that can quickly go through the full document store and pass a set of candidate documents to the Reader. It is an tool for sifting out the obvious negative cases, saving the Reader from doing more work than it needs to and speeding up the querying process. There are two fundamentally different categories of retrievers: sparse (e.g. TF-IDF, BM25) and dense (e.g. DPR, sentence-transformers).

**Available Options**  
- DensePassageRetriever
- ElasticsearchRetriever
- EmbeddingRetriever
- TfidfRetriever

**Example**  

```python
retriever = DensePassageRetriever(document_store=document_store,
                                  query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                  passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                  use_gpu=True,
                                  batch_size=16,
                                  embed_title=True)
retriever.retrieve(query="Why did the revenue increase?")
# returns: [Document, Document]
```

-> See [docs](https://haystack.deepset.ai/docs/latest/retrievermd) for details


### 5) Readers

**What**  
Neural networks (i.e. mostly Transformer-based) that read through texts in detail to find an answer. Use diverse models like BERT, RoBERTa or
XLNet trained via [FARM](https://github.com/deepset-ai/FARM) or on SQuAD-like datasets. The Reader takes multiple passages of text as input
and returns top-n answers with corresponding confidence scores. Both readers can load either a local model or any public model from [Hugging
Face's model hub](https://huggingface.co/models)

**Available Options**   
- FARMReader: Reader based on [FARM](https://github.com/deepset-ai/FARM) incl. extensive configuration options and speed optimizations
- TransformersReader: Reader based on the `pipeline` class of HuggingFace's [Transformers](https://github.com/huggingface/transformers).  
**Both** Readers can load models directly from HuggingFace's model hub.

**Example**  

```python
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2",
                use_gpu=False, no_ans_boost=-10, context_window_size=500,
                top_k_per_candidate=3, top_k_per_sample=1,
                num_processes=8, max_seq_len=256, doc_stride=128)

# Optional: Training & eval
reader.train(...)
reader.eval(...)

# Predict
reader.predict(question="Who is the father of Arya Starck?", documents=documents, top_k=3)
```
-> See [docs](https://haystack.deepset.ai/docs/latest/readermd) for details

### 6) Pipelines

**What**  
In order to build modern search pipelines, you need two things: powerful building blocks and a flexible way to stick them together.
The `Pipeline` class is exactly build for this purpose and enables many search scenarios beyond QA. The core idea: you can build a Directed Acyclic Graph (DAG) where each node is one "building block" (Reader, Retriever, Generator ...).

**Available Options**   
- Standard nodes: Reader, Retriever, Generator ...
- Join nodes: For example, combine results of multiple retrievers via the `JoinDocuments` node
- Decision Nodes: For example, classify an incoming query and depending on the results execute only certain branch of your graph 

**Example**  
A minimal Open-Domain QA Pipeline:

```python
p = Pipeline()
p.add_node(component=retriever, name="ESRetriever1", inputs=["Query"])
p.add_node(component=reader, name="QAReader", inputs=["ESRetriever1"])
res = p.run(query="What did Einstein work on?", top_k_retriever=1)

```
You can **draw the DAG** to better inspect what you are building:
```python
p.draw(path="custom_pipe.png")
```
![image](https://user-images.githubusercontent.com/1563902/102451716-54813700-4039-11eb-881e-f3c01b47ca15.png)

-> See [docs](https://haystack.deepset.ai/docs/latest/pipelinesmd) for details and example of more complex pipelines

### 7) REST API
**What**  
A simple REST API based on [FastAPI](https://fastapi.tiangolo.com/) is provided to:

-   search answers in texts ([extractive QA](https://github.com/deepset-ai/haystack/blob/master/rest_api/controller/search.py))
-   search answers by comparing user question to existing questions
    ([FAQ-style QA](https://github.com/deepset-ai/haystack/blob/master/rest_api/controller/search.py))
-   collect & export user feedback on answers to gain domain-specific
    training data
    ([feedback](https://github.com/deepset-ai/haystack/blob/master/rest_api/controller/feedback.py))
-   allow basic monitoring of requests (currently via APM in Kibana)

**Example**  
To serve the API, adjust the values in `rest_api/config.py` and run:

    gunicorn rest_api.application:app -b 0.0.0.0:8000 -k uvicorn.workers.UvicornWorker -t 300

You will find the Swagger API documentation at
<http://127.0.0.1:8000/docs>

### 8) Labeling Tool

-   Use the [hosted version](https://annotate.deepset.ai/login) (Beta) or deploy it yourself with the [Docker Images](https://github.com/deepset-ai/haystack/blob/master/annotation_tool).
-   Create labels with different techniques: Come up with questions (+ answers) while reading passages (SQuAD style) or have a set of predefined questions and look for answers in the document (~ Natural Questions).
-   Structure your work via organizations, projects, users
-   Upload your documents or import labels from an existing SQuAD-style dataset

![image](https://raw.githubusercontent.com/deepset-ai/haystack/master/docs/_src/img/annotation_tool.png)


## :heart: Contributing

We are very open to contributions from the community - be it the fix of a small typo or a completely new feature! You don't need to be an
Haystack expert for providing meaningful improvements. To avoid any extra work on either side, please check our [Contributor Guidelines](https://github.com/deepset-ai/haystack/blob/master/CONTRIBUTING.md) first.

We'd also like to invite you to our Slack community channels. Please join [here](https://haystack.deepset.ai/community/join)!

Tests will automatically run for every commit you push to your PR. You can also run them locally by executing [pytest](https://docs.pytest.org/en/stable/) in your terminal from the root folder of this repository:

All tests:
``` bash
cd test
pytest
```

You can also only run a subset of tests by specifying a marker and the optional "not" keyword: 
``` bash
cd test
pytest -m not elasticsearch
pytest -m elasticsearch
pytest -m generator
pytest -m tika
pytest -m not slow
...
```
