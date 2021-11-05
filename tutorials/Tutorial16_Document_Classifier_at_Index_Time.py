"""
Extending your Metadata using DocumentClassifiers at Index Time

DocumentClassifier adds the classification result (label and score) to Document's meta property. 
Hence, we can use it to classify documents at index time. The result can be accessed at query time: for example by applying a filter for "classification.label".

This tutorial will show you how to integrate a classification model into your preprocessing steps and how you can filter for this additional metadata at query time.
"""

# Here are the imports we need
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes import PreProcessor, TransformersDocumentClassifier, FARMReader, ElasticsearchRetriever
from haystack.schema import Document
from haystack.utils import convert_files_to_dicts, fetch_archive_from_http, print_answers, launch_es


def tutorial16_document_classifier_at_index_time():
    # This fetches some sample files to work with

    doc_dir = "data/preprocessing_tutorial"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/preprocessing_tutorial.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # read and preprocess documents
    # note that you can also use the document classifier before applying the PreProcessor, e.g. before splitting your documents
    all_docs = convert_files_to_dicts(dir_path="data/preprocessing_tutorial")
    preprocessor_sliding_window = PreProcessor(
        split_overlap=3,
        split_length=10,
        split_respect_sentence_boundary=False
    )
    docs_sliding_window = preprocessor_sliding_window.process(all_docs)

    """ 
    
    ## DocumentClassifier

    We can enrich the document metadata at index time using any transformers document classifier model.
    Here we use an emotion model that distinguishes between 'sadness', 'joy', 'love', 'anger', 'fear' and 'surprise'.
    These classes can later on be accessed at query time.

    """

    doc_classifier_model = 'bhadresh-savani/distilbert-base-uncased-emotion'
    doc_classifier = TransformersDocumentClassifier(model_name_or_path=doc_classifier_model, batch_size=16)

    # convert to Document using a fieldmap for custom content fields the classification should run on
    field_map = {}
    docs_to_classify = [Document.from_dict(d, field_map=field_map) for d in docs_sliding_window]

    # classify using gpu, batch_size makes sure we do not run out of memory
    classified_docs = doc_classifier.predict(docs_to_classify)

    # convert back to dicts if you want, note that DocumentStore.write_documents() can handle Documents too
    # classified_docs = [doc.to_dict(field_map=field_map) for doc in classified_docs]
    
    # let's see how it looks: there should be a classification result in the meta entry containing label and score.
    print(classified_docs[0].to_dict(field_map=field_map))

    launch_es()

    # Connect to Elasticsearch
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

    # Now, let's write the docs to our DB.
    document_store.write_documents(classified_docs)

    # check if indexed docs contain classification results
    test_doc = document_store.get_all_documents()[0]
    print(f'document {test_doc.id} has label {test_doc.meta["classification"]["label"]}')

    # Initialize QA-Pipeline
    from haystack.pipelines import ExtractiveQAPipeline
    retriever = ElasticsearchRetriever(document_store=document_store)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    pipe = ExtractiveQAPipeline(reader, retriever)    
    
    ## Voilà! Ask a question while filtering for "joy"-only documents
    prediction = pipe.run(
        query="How is heavy metal?", params={"Retriever": {"top_k": 10, "filters": {"classification.label": ["joy"]}}, "Reader": {"top_k": 5}}
    )

    print_answers(prediction, details="high")

if __name__ == "__main__":
    tutorial16_document_classifier_at_index_time()

# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/
