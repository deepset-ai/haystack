from typing import List

import json
from haystack import BaseComponent, Document
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
from transformers import pipeline


class EntityExtractor(BaseComponent):
    outgoing_edges = 1

    def __init__(self,
                 model_name_or_path="dslim/bert-base-NER"):
        
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        token_classifier = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        self.model = pipeline("ner", model=token_classifier, tokenizer=tokenizer)#, aggregation_strategy="simple")

    def run(self, documents: List[Document]):
        for doc in documents:
            doc.meta["entities"] = self.extract(doc.text)
        output = {"documents": documents}
        return output, "output_1"

    def extract(self, text):
        entities = self.model(text)
        return entities


def print_ner_and_qa(output): 
    """
    [
        { 
            answer: { ... }
            entities: [ { ... }, {} ]
        }
    ]
    """
    compact_output = []
    for answer in output["answers"]:

        entities = []
        for entity in answer["meta"]["entities"]:
            if entity["start"] >= answer["offset_start_in_doc"] and entity["end"] <= answer["offset_end_in_doc"]]:
                entities.append(entity["word"])  

        compact_output.append({
            "answer": answer["answer"],
            "entities": entities
        })
        
    print(json.dumps(compact_output, indent=4, default=str))
