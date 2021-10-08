from typing import List, Union, Dict

from haystack import BaseComponent, Document
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
from transformers import pipeline


class EntityExtractor(BaseComponent):
    outgoing_edges = 1

    def __init__(self,
                 model_name_or_path="dslim/bert-base-NER"):
        
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        token_classifier = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        self.model = pipeline("ner", model=token_classifier, tokenizer=tokenizer, aggregation_strategy="simple")

    def run(self, documents: Union[List[Document], List[Dict]]):
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
        entities = self.model(text)
        return entities

def print_ner_and_qa(self, output): 
    """
    [
        { 
            answer: { ... }
            entities: [ { ... }, {} ]
        }
    ]
    """
    pass