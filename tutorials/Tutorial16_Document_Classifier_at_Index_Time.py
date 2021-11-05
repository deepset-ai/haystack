"""
Preprocessing

Haystack includes a suite of tools to extract text from different file types, normalize white space
and split text into smaller pieces to optimize retrieval.
These data preprocessing steps can have a big impact on the systems performance and effective handling of data is key to getting the most out of Haystack.

Ultimately, Haystack pipelines expect data to be provided as a list documents in the following dictionary format:

docs = [
    {
        'text': DOCUMENT_TEXT_HERE,
        'meta': {'name': DOCUMENT_NAME, ...}
    }, ...
]

This tutorial will show you all the tools that Haystack provides to help you cast your data into the right format.
"""

# Here are the imports we need
from haystack.nodes import PreProcessor, TransformersDocumentClassifier
from haystack.schema import Document
from haystack.utils import convert_files_to_dicts, fetch_archive_from_http


def tutorial16_document_classifier_at_index_time():
    # This fetches some sample files to work with

    doc_dir = "data/preprocessing_tutorial"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/preprocessing_tutorial.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    """
    ## Converters
    
    Haystack's converter classes are designed to help you turn files on your computer into the documents
    that can be processed by the Haystack pipeline.
    There are file converters for txt, pdf, docx files as well as a converter that is powered by Apache Tika.
    The parameter `valid_langugages` does not convert files to the target language, but checks if the conversion worked as expected.
    For converting PDFs, try changing the encoding to UTF-8 if the conversion isn't great.
    """
    # Haystack also has a convenience function that will automatically apply the right converter to each file in a directory.

    all_docs = convert_files_to_dicts(dir_path="data/preprocessing_tutorial")

    """
    
    ## PreProcessor
    
    The PreProcessor class is designed to help you clean text and split text into sensible units.
    File splitting can have a very significant impact on the system's performance.
    Have a look at the [Preprocessing](https://haystack.deepset.ai/docs/latest/preprocessingmd)
    and [Optimization](https://haystack.deepset.ai/docs/latest/optimizationmd) pages on our website for more details.
    """

    """
    A commonly used strategy to split long documents, especially in the field of Question Answering,
    is the sliding window approach. If `split_length=10` and `split_overlap=3`, your documents will look like this:
    
    - doc1 = words[0:10]
    - doc2 = words[7:17]
    - doc3 = words[14:24]
    - ...
    
    You can use this strategy by following the code below.
    """

    # Sliding window approach

    preprocessor_sliding_window = PreProcessor(
        split_overlap=3,
        split_length=10,
        split_respect_sentence_boundary=False
    )
    docs_sliding_window = preprocessor_sliding_window.process(all_docs)

    doc1 = docs_sliding_window[0]["content"][:200]
    doc2 = docs_sliding_window[1]["content"][:100]
    doc3 = docs_sliding_window[2]["content"][:100]

    print("Document 1: \"" + doc1 + "...\"")
    print("Document 2: \"" + doc2 + "...\"")
    print("Document 3: \"" + doc3 + "...\"")

    """ 
    
    ## DocumentClassifier

    We can enrich the document metadata at index time using any transformers document classifier model.
    Here we use an emotion model that distinguishes between 'sadness', 'joy', 'love', 'anger', 'fear' and 'surprise'.
    These classes can later on be accessed at query time.

    """

    doc_classifier_model = 'bhadresh-savani/distilbert-base-uncased-emotion'
    doc_classifier = TransformersDocumentClassifier(model_name_or_path=doc_classifier_model)

    # convert to Document using a fieldmap for custom content fields the classification should run on
    field_map = {}
    docs_to_classify = [Document.from_dict(d, field_map=field_map) for d in docs_sliding_window]
    classified_docs = doc_classifier.predict(docs_to_classify)

    # convert back to dicts if you want, note that DocumentStore.write_documents() can handle Documents too
    classified_docs = [doc.to_dict(field_map=field_map) for doc in classified_docs]
    print(classified_docs[0])

if __name__ == "__main__":
    tutorial16_document_classifier_at_index_time()

# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/
