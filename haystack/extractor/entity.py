from typing import List

from haystack import BaseComponent, Document
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
from transformers import pipeline


class EntityExtractor(BaseComponent):
    outgoing_edges = 1

    def __init__(self,
                 model_name_or_path="dslim/bert-base-NER"):
        
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        token_classifier = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        self.model = pipeline("ner", model=token_classifier, tokenizer=tokenizer)

    def run(self, documents: List[Document]):
        for doc in documents:
            doc.meta["entities"] = self.extract(doc.text)
            for entity in doc.meta["entities"]:
                del entity["index"]
        output = {"documents": documents}
        return output, "output_1"

    def extract(self, text):
        entities = self.model(text)
        combined_entities = self.combine_ner_words(entities)
        return combined_entities

    def combine_ner_words(self, entities):
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