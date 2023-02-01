- Start Date: 2022-11-28
- Proposal PR: [#3638](https://github.com/deepset-ai/haystack/issues/3638)
- Github Issue: [#3550](https://github.com/deepset-ai/haystack/issues/3550)

# Summary

Using Haystack for questions answering pipelines is prettier easy, but most of the time users have CSV files containing
their knowledge base with questions and there associated answers.
Unfortunately there is no easy way to dynamically update the knowledge base or import new data from CSV though rest API
using YAML, as there are no CSV parser.

Having a basic way to dynamically index a CSV file always requires development of a new nodes.

# Basic example

To define an FAQ query and **indexing** pipeline we would then simply do :
```yaml
# To allow your IDE to autocomplete and validate your YAML pipelines, name them as <name of your choice>.haystack-pipeline.yml

version: ignore

components:    # define all the building-blocks for Pipeline
  - name: DocumentStore
    type: ElasticsearchDocumentStore
    params:
      host: localhost
      embedding_field: question_emb
      embedding_dim: 384
      excluded_meta_data:
        - question_emb
      similarity: cosine
  - name: Retriever
    type: EmbeddingRetriever
    params:
      document_store: DocumentStore    # params can reference other components defined in the YAML
      embedding_model: sentence-transformers/all-MiniLM-L6-v2
      scale_score: False
  - name: CSVConverter
    type: CsvTextConverter

pipelines:
  - name: indexing
    nodes:
      - name: CSVConverter
        inputs: [File]
      - name: Retriever
        inputs: [ CSVConverter ]
      - name: DocumentStore
        inputs: [ Retriever ]
```

# Motivation

Using YAML pipeline description it's not possible to describe a CSV FAQ indexing pipeline that takes CSV files as input
containing questions and answers and index them. It's a basic usage that still requires coding.

As we are presenting a tutorial `Utilizing Existing FAQs for Question Answering` it would be great to have this basic
node so that anyone can quickly run an FAQ Question Answering pipeline using only a YAML description and import their
CSV though REST API.

# Detailed design

I've added a new node: **`CsvTextConverter`** . It takes a file input, parse it as FAQ CSV file having `question` and `answer` column
and outputs `Document`s.

For now the node is very simple: can only handle a fixed CSV format and no other tabular data. It also
can't produce documents that are not of type `text`. These shortcomings can be addressed in later enhancements.

# Drawbacks

We could consider that developing this custom node is easy and a good way to learn Haystack,
but casual users shouldn't need to know this much before being able to index CSV files.

# Alternatives

Didn't consider any other design.

# Adoption strategy

It doesn't introduce any breaking change, any users having FAQs pipeline would be able to use the official nodes instead
of their existing ones.

# How we teach this

This may require updating this tutorial [Utilizing Existing FAQs for Question Answering](https://haystack.deepset.ai/tutorials/04_faq_style_qa)
and to document those 2 nodes.
