from typing import List

from haystack.nodes.base import BaseComponent
from haystack.schema import Document


class Doc2QueryExpander(BaseComponent):
    """
    Node that allows to add questions generated from a QuestionGenerator instance to a specified document meta field.
    This node may be used if you want to implement a doc2query-type pipeline.
    See: https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf
    """
    outgoing_edges = 1

    def __init__(self, target_field: str):
        """
        Initializing a Doc2QueryExpander instance.
        :param target_field: Name of the metadata field where the generated questions should be stored.
        """
        self.set_config(target_field=target_field)
        self.target_field = target_field

    def run(self, documents: List[Document], generated_questions: List[str]):
        for questions, document in zip(generated_questions, documents):
            document.meta[self.target_field] = " ".join(questions["questions"])

        docs = [doc.to_dict() for doc in documents]
        outputs = {"documents": docs}

        return outputs, "output_1"
