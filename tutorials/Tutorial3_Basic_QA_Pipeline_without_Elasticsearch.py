# ## Task: Build a Question Answering pipeline without Elasticsearch
#
# Haystack provides alternatives to Elasticsearch for developing quick prototypes.
#
# You can use an `InMemoryDocumentStore` or a `SQLDocumentStore`(with SQLite) as the document store.
#
# If you are interested in more feature-rich Elasticsearch, then please refer to the Tutorial 1.

from haystack.document_stores import InMemoryDocumentStore, SQLDocumentStore
from haystack.nodes import FARMReader, TransformersReader, TfidfRetriever
from haystack.utils import clean_wiki_text, convert_files_to_dicts, fetch_archive_from_http, print_answers


def tutorial3_basic_qa_pipeline_without_elasticsearch():
    # In-Memory Document Store
    document_store = InMemoryDocumentStore()

    # or, alternatively, SQLite Document Store
    # document_store = SQLDocumentStore(url="sqlite:///qa.db")


    # ## Preprocessing of documents
    #
    # Haystack provides a customizable pipeline for:
    # - converting files into texts
    # - cleaning texts
    # - splitting texts
    # - writing them to a Document Store

    # In this tutorial, we download Wikipedia articles on Game of Thrones, apply a basic cleaning function, and index
    # them in Elasticsearch.
    # Let's first get some documents that we want to query
    # Here: 517 Wikipedia articles for Game of Thrones
    doc_dir = "data/article_txt_got"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # convert files to dicts containing documents that can be indexed to our datastore
    dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
    # You can optionally supply a cleaning function that is applied to each doc (e.g. to remove footers)
    # It must take a str as input, and return a str.

    # Now, let's write the docs to our DB.
    document_store.write_documents(dicts)


    # ## Initalize Retriever, Reader & Pipeline
    #
    # ### Retriever
    #
    # Retrievers help narrowing down the scope for the Reader to smaller units of text where
    # a given question could be answered.
    #
    # With InMemoryDocumentStore or SQLDocumentStore, you can use the TfidfRetriever. For more
    # retrievers, please refer to the tutorial-1.

    # An in-memory TfidfRetriever based on Pandas dataframes
    retriever = TfidfRetriever(document_store=document_store)

    # ### Reader
    #
    # A Reader scans the texts returned by retrievers in detail and extracts the k best answers. They are based
    # on powerful, but slower deep learning models.
    #
    # Haystack currently supports Readers based on the frameworks FARM and Transformers.
    # With both you can either load a local model or one from Hugging Face's model hub (https://huggingface.co/models).

    # **Here:**                   a medium sized RoBERTa QA model using a Reader based on
    #                             FARM (https://huggingface.co/deepset/roberta-base-squad2)
    # **Alternatives (Reader):**  TransformersReader (leveraging the `pipeline` of the Transformers package)
    # **Alternatives (Models):**  e.g. "distilbert-base-uncased-distilled-squad" (fast) or
    #                             "deepset/bert-large-uncased-whole-word-masking-squad2" (good accuracy)
    # **Hint:**                   You can adjust the model to return "no answer possible" with the no_ans_boost.
    #                             Higher values mean the model prefers "no answer possible".

    # #### FARMReader
    #
    # Load a  local model or any of the QA models on
    # Hugging Face's model hub (https://huggingface.co/models)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)


    # #### TransformersReader
    # Alternative:
    # reader = TransformersReader(model_name_or_path="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased", use_gpu=-1)

    # ### Pipeline
    #
    # With a Haystack `Pipeline` you can stick together your building blocks to a search pipeline.
    # Under the hood, `Pipelines` are Directed Acyclic Graphs (DAGs) that you can easily customize for your own use cases.
    # To speed things up, Haystack also comes with a few predefined Pipelines. One of them is the `ExtractiveQAPipeline` that combines a retriever and a reader to answer our questions.
    # You can learn more about `Pipelines` in the [docs](https://haystack.deepset.ai/docs/latest/pipelinesmd).
    from haystack.pipelines import ExtractiveQAPipeline
    pipe = ExtractiveQAPipeline(reader, retriever)

    ## Voilà! Ask a question!
    prediction = pipe.run(
        query="Who is the father of Arya Stark?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
    )

    # prediction = pipe.run(query="Who created the Dothraki vocabulary?", params={"Reader": {"top_k": 5}})
    # prediction = pipe.run(query="Who is the sister of Sansa?", params={"Reader": {"top_k": 5}})

    # Now you can either print the object directly
    print("\n\nRaw object:\n")
    from pprint import pprint
    pprint(prediction)

    # Sample output:    
    # {
    #     'answers': [ <Answer: answer='Eddard', type='extractive', score=0.9919578731060028, offsets_in_document=[{'start': 608, 'end': 615}], offsets_in_context=[{'start': 72, 'end': 79}], document_id='cc75f739897ecbf8c14657b13dda890e', meta={'name': '454_Music_of_Game_of_Thrones.txt'}}, context='...' >,
    #                  <Answer: answer='Ned', type='extractive', score=0.9767240881919861, offsets_in_document=[{'start': 3687, 'end': 3801}], offsets_in_context=[{'start': 18, 'end': 132}], document_id='9acf17ec9083c4022f69eb4a37187080', meta={'name': '454_Music_of_Game_of_Thrones.txt'}}, context='...' >,
    #                  ...
    #                ]
    #     'documents': [ <Document: content_type='text', score=0.8034909798951382, meta={'name': '332_Sansa_Stark.txt'}, embedding=None, id=d1f36ec7170e4c46cde65787fe125dfe', content='\n===\'\'A Game of Thrones\'\'===\nSansa Stark begins the novel by being betrothed to Crown ...'>,
    #                    <Document: content_type='text', score=0.8002150354529785, meta={'name': '191_Gendry.txt'}, embedding=None, id='dd4e070a22896afa81748d6510006d2', 'content='\n===Season 2===\nGendry travels North with Yoren and other Night's Watch recruits, including Arya ...'>,
    #                    ...
    #                  ],
    #     'no_ans_gap':  11.688868522644043,
    #     'node_id': 'Reader',
    #     'params': {'Reader': {'top_k': 5}, 'Retriever': {'top_k': 5}},
    #     'query': 'Who is the father of Arya Stark?',
    #     'root_node': 'Query'
    # }

    # Note that the documents contained in the above object are the documents filtered by the Retriever from
    # the document store. Although the answers were extracted from these documents, it's possible that many
    # answers were taken from a single one of them, and that some of the documents were not source of any answer.

    # Or use a util to simplify the output
    # Change `minimum` to `medium` or `all` to raise the level of detail
    print("\n\nSimplified output:\n")
    print_answers(prediction, details="minimum")


if __name__ == "__main__":
    tutorial3_basic_qa_pipeline_without_elasticsearch()

# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/