Glossary
========

**BERT** - A popular, transformer based language model which has been improved upon but is still considered a common benchmark.

**Deep Learning** - A subfield of AI that focuses on building effective model architectures composed of stacked Neural Network layers.

**Dense** - Vectors that contain many non-zero values are considered dense.
Retrieval methods can also be called dense if they create dense vector representations of documents.

**Document Store** - The component in Haystack that stores the text documents and their metadata.
Can have a variety of backends such as Elasticsearch, SQL or FAISS.

**FARM** - An open-source transfer learning `framework <https://github.com/deepset-ai/FARM>`_ by deepset.
FARM's question answering models are used in Haystack's Readers.

**Finder** - The component in Haystack that connects the Retriever to the Reader.

**Indexing** - To store data in a database in a way that optimises retrieval time.
The exact steps involved in indexing depend on what kind of retrieval method is chosen.

**Language Model** - The component in an NLP model that stores general language understanding, but no task specific knowledge.

**Model Hub** - The repository set up by HuggingFace where trained models can be saved to and loaded from.
With Haystack, you can directly load and use any question answering model found on the model hub.

**Neural Network** - A machine learning architecture composed of artificial neurons that learn a task when exposed to labelled training data.

**Querying** - The task of returning relevant documents from a database.

**Question Answering** - A popular task in the world of NLP where systems have to find answers to questions.
The term is generally used to refer to extractive question answering,
where a system has to find the minimal text span in a given document that contains the answer to the question.

**Reader** - The component in Haystack that does the closest reading of a document to extract
the exact text which answers a question.
It is, at its core, a trained Question Answering model.

**Retriever** - A lightweight filter that selects only the most relevant documents for the Reader to further process.

**Semantic Search** - A style of search that relies not on the matching of exact string forms
but on the similarity of meaning between a query and a piece of text.

**Sparse** - Vectors that are composed primarily of zeros are called sparse.
Retrieval methods are also considered sparse if they build sparse vector representations of documents.

**Transformers** - Originally refers to the deep learning architecture that is composed of stacked self-attention layers
(first conceptualised `here <https://arxiv.org/pdf/1706.03762.pdf>`_).
Can also refer to HuggingFace's `repository <https://github.com/huggingface/transformers>`_
which contains implementations of popular model architectures.
















Prediction Head


