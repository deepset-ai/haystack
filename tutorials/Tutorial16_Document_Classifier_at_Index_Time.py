# # Extending your Metadata using DocumentClassifiers at Index Time
# 
# With DocumentClassifier it's possible to automatically enrich your documents with categories, sentiments, topics or whatever metadata you like. This metadata could be used for efficient filtering or further processing. Say you have some categories your users typically filter on. If the documents are tagged manually with these categories, you could automate this process by training a model or using zero shot classification. This tutorial shows how to integrate it in your indexing pipeline.

# DocumentClassifier adds the classification result (label and score) to Document's meta property.
# Hence, we can use it to classify documents at index time. \
# The result can be accessed at query time: for example by applying a filter for "classification.label".

# This tutorial will show you how to integrate a classification model into your preprocessing steps and how you can filter for this additional metadata at query time. In the last section we show how to put it all together and create an indexing pipeline.


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


    # ## Read and preprocess documents

    # note that you can also use the document classifier before applying the PreProcessor, e.g. before splitting your documents
    all_docs = convert_files_to_dicts(dir_path=doc_dir)
    preprocessor_sliding_window = PreProcessor(
        split_overlap=3,
        split_length=10,
        split_respect_sentence_boundary=False
    )
    docs_sliding_window = preprocessor_sliding_window.process(all_docs)

    # ## Apply DocumentClassifier

    # We can enrich the document metadata at index time using any transformers document classifier model.
    # Here we use a zero shot model that is supposed to classify our documents in 'music', 'natural language processing' and 'history'.
    # These classes can later on be accessed at query time.

    doc_classifier = TransformersDocumentClassifier(model_name_or_path="cross-encoder/nli-distilroberta-base",
        task="zero-shot-classification",
        labels=["music", "natural language processing", "history"],
        batch_size=16, use_gpu=-1
    )

    # we can also use any other transformers model besides zero shot classification

    # doc_classifier_model = 'bhadresh-savani/distilbert-base-uncased-emotion'
    # doc_classifier = TransformersDocumentClassifier(model_name_or_path=doc_classifier_model, batch_size=16, use_gpu=-1)

    # we could also specifiy a different field we want to run the classification on

    # doc_classifier = TransformersDocumentClassifier(model_name_or_path="cross-encoder/nli-distilroberta-base",
    #    task="zero-shot-classification",
    #    labels=["music", "natural language processing", "history"],
    #    batch_size=16, use_gpu=-1,
    #    classification_field="description")


    # convert to Document using a fieldmap for custom content fields the classification should run on
    docs_to_classify = [Document.from_dict(d) for d in docs_sliding_window]

    # classify using gpu, batch_size makes sure we do not run out of memory
    classified_docs = doc_classifier.predict(docs_to_classify)


    # let's see how it looks: there should be a classification result in the meta entry containing labels and scores.
    print(classified_docs[0].to_dict())


    # ## Indexing

    launch_es()

    # Connect to Elasticsearch
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

    # Now, let's write the docs to our DB.
    document_store.delete_all_documents()
    document_store.write_documents(classified_docs)

    # check if indexed docs contain classification results
    test_doc = document_store.get_all_documents()[0]
    print(f'document {test_doc.id} with content \n\n{test_doc.content}\n\nhas label {test_doc.meta["classification"]["label"]}')


    # ## Querying the data

    # All we have to do to filter for one of our classes is to set a filter on "classification.label".

    # Initialize QA-Pipeline
    from haystack.pipelines import ExtractiveQAPipeline
    retriever = ElasticsearchRetriever(document_store=document_store)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
    pipe = ExtractiveQAPipeline(reader, retriever)    

    ## Voilà! Ask a question while filtering for "music"-only documents
    prediction = pipe.run(
        query="What is heavy metal?", params={"Retriever": {"top_k": 10, "filters": {"classification.label": ["music"]}}, "Reader": {"top_k": 5}}
    )

    print_answers(prediction, details="high")


    # ## Wrapping it up in an indexing pipeline

    from pathlib import Path
    from haystack.pipelines import Pipeline
    from haystack.nodes import TextConverter, FileTypeClassifier, PDFToTextConverter, DocxToTextConverter

    # we need to set the batch_size so each batch fits into memory
    index_time_document_classifier = TransformersDocumentClassifier(
        model_name_or_path="cross-encoder/nli-distilroberta-base",
        task="zero-shot-classification",
        labels=["music", "natural language processing", "history"], 
        batch_size=16, use_gpu=-1
    )

    file_type_classifier = FileTypeClassifier()
    text_converter = TextConverter()
    pdf_converter = PDFToTextConverter()
    docx_converter = DocxToTextConverter()

    indexing_pipeline_with_classification = Pipeline()
    indexing_pipeline_with_classification.add_node(component=file_type_classifier, name="FileTypeClassifier", inputs=["File"])
    indexing_pipeline_with_classification.add_node(component=text_converter, name="TextConverter", inputs=["FileTypeClassifier.output_1"])
    indexing_pipeline_with_classification.add_node(component=pdf_converter, name="PdfConverter", inputs=["FileTypeClassifier.output_2"])
    indexing_pipeline_with_classification.add_node(component=docx_converter, name="DocxConverter", inputs=["FileTypeClassifier.output_4"])
    indexing_pipeline_with_classification.add_node(component=preprocessor_sliding_window, name="Preprocessor", inputs=["TextConverter", "PdfConverter", "DocxConverter"])
    indexing_pipeline_with_classification.add_node(component=index_time_document_classifier, name="IndexTimeDocumentClassifier", inputs=["Preprocessor"])
    indexing_pipeline_with_classification.add_node(component=retriever, name="Retriever", inputs=["IndexTimeDocumentClassifier"])
    indexing_pipeline_with_classification.add_node(component=document_store, name="DocumentStore", inputs=["Retriever"])
    indexing_pipeline_with_classification.draw("index_time_document_classifier.png")

    document_store.delete_documents()
    txt_files = [f for f in Path(doc_dir).iterdir() if f.suffix == '.txt']
    pdf_files = [f for f in Path(doc_dir).iterdir() if f.suffix == '.pdf']
    docx_files = [f for f in Path(doc_dir).iterdir() if f.suffix == '.docx']
    indexing_pipeline_with_classification.run(file_paths=txt_files)
    indexing_pipeline_with_classification.run(file_paths=pdf_files)
    indexing_pipeline_with_classification.run(file_paths=docx_files)

    document_store.get_all_documents()[0]

    # we can store this pipeline and use it from the REST-API
    indexing_pipeline_with_classification.save_to_yaml("indexing_pipeline_with_classification.yaml")


if __name__ == "__main__":
    tutorial16_document_classifier_at_index_time()

# ## About us
# 
# This [Haystack](https://github.com/deepset-ai/haystack/) notebook was made with love by [deepset](https://deepset.ai/) in Berlin, Germany
# 
# We bring NLP to the industry via open source!  
# Our focus: Industry specific language models & large scale QA systems.  
#   
# Some of our other work: 
# - [German BERT](https://deepset.ai/german-bert)
# - [GermanQuAD and GermanDPR](https://deepset.ai/germanquad)
# - [FARM](https://github.com/deepset-ai/FARM)
# 
# Get in touch:
# [Twitter](https://twitter.com/deepset_ai) | [LinkedIn](https://www.linkedin.com/company/deepset-ai/) | [Slack](https://haystack.deepset.ai/community/join) | [GitHub Discussions](https://github.com/deepset-ai/haystack/discussions) | [Website](https://deepset.ai)
# 
# By the way: [we're hiring!](https://www.deepset.ai/jobs)
# 
