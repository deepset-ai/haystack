from typing import List, Union, Dict

from haystack import BaseComponent, Document
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
from transformers import pipeline


class EntityExtractor(BaseComponent):
    """
    This node is used to extract entities out of documents.
    The most common use case for this would be as a named entity extractor.
    The default model used is dslim/bert-base-NER.
    This node can be placed in a querying pipeline to perform entity extraction on retrieved documents only,
    or it can be placed in an indexing pipeline so that all documents in the document store have extracted entities.
    The entities extracted by this Node will populate Document.entities
    """

    outgoing_edges = 1

    def __init__(self,
                 model_name_or_path="dslim/bert-base-NER"):
        
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        token_classifier = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        self.model = pipeline("ner", model=token_classifier, tokenizer=tokenizer, aggregation_strategy="simple")

    def run(self, documents: Union[List[Document], List[Dict]]):
        """
        This is the method called when this node is used in a pipeline
        """
        for doc in documents:
            # In a querying pipeline, doc is a haystack.schema.Document object
            try:
                doc.meta["entities"] = self.extract(doc.text)
            # In an indexing pipeline, doc is a dictionary
            except AttributeError:
                doc["meta"]["entities"] = self.extract(doc["text"])
        output = {"documents": documents}
        return output, "output_1"

    def extract(self, text):
        """
        This function can be called to perform entity extraction when using the node in isolation.
        """
        entities = self.model(text)
        return entities

def print_ner_and_qa(self, output):
    """
    This is a utility function used to print QA answer prediction and the extracted entities that occur within these answers.
    """
    """
    [
        { 
            answer: { ... }
            entities: [ { ... }, {} ]
        }
    ]
    """
    pass