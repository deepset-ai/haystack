from typing import Optional, Set
from spacy.tokens import Doc
from operator import itemgetter
from enum import Enum


class QuestionType(Enum):
    BooleanQuestion = "boolean_question"
    CountQuestion = "count_question"
    ListQuestion = "list_question"


class Question:

    def __init__(self, question_text: str):
        self.question_text: str = question_text
        self.question_type: Optional[QuestionType] = None
        self.doc: Optional[Doc] = None
        self.entities: Optional[Set[str]] = None
        self.relations: Optional[Set[str]] = None

    def analyze(self, nlp, entity_to_frequency, alias_to_entity_and_prob, top_relations) -> (Optional[Set[str]], Optional[Set[str]], Optional[QuestionType]):
        self.doc = nlp(self.question_text)
        self.entities = self.entity_linking(entity_to_frequency=entity_to_frequency, alias_to_entity_and_prob=alias_to_entity_and_prob)
        self.relations = self.relation_linking(top_relations=top_relations, nlp=nlp)
        self.question_type = self.classify_type()
        return self.entities, self.relations, self.question_type

    def classify_type(self):
        """
        Classify question into one of three classes based on a heuristic:
        1. count question -> answer: number
        2. boolean question -> answer: boolean
        3. list question -> answer: list of resources
        """
        lemmas = [token.lemma_ for token in self.doc]
        if lemmas[0] == "how":
            return QuestionType.CountQuestion

        pos_tags = [token.pos_ for token in self.doc]
        if pos_tags[0] == "AUX":
            # e.g., question starts with "Does", "Is", "Are" or "Was"
            return QuestionType.BooleanQuestion

        return QuestionType.ListQuestion

    def entity_linking(self, entity_to_frequency, alias_to_entity_and_prob):
        """
                Link spacy entities mentioned in a question to entities that exist in our knowledge base
                """
        entities = set()
        if self.doc.ents:  # if spacy recognized any entities
            for ent in self.doc.ents:
                if ent.text in entity_to_frequency:
                    entities.add(ent.text)
                elif ent.text in alias_to_entity_and_prob:
                    entities.add(max(alias_to_entity_and_prob[ent.text], key=itemgetter(1))[0])
        else:  # else try to link nouns to entities
            for token in self.doc:
                if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                    if token.text in entity_to_frequency:
                        entities.add(token.text)
                    elif token.text in alias_to_entity_and_prob:
                        entities.add(max(alias_to_entity_and_prob[token.text], key=itemgetter(1))[0])

        # TODO remove debug hack
        entities = {"Albus Dumbledore"}
        entities = {f"<https://deepset.ai/harry_potter/{entity.replace(' ','_')}>" for entity in entities}
        return entities

    def relation_linking(self, top_relations, nlp):
        """
                Link verbs and nouns mentioned in a question to relations that exist in our knowledge base
                """
        # todo train with distant supervision
        #  "Distant supervision [62]: "We learn indicator words for each relation using text from Wikipedia where entity mentions were identified.
        #  This allows deriving noisy training examples: A sentence expresses relation r if it contains two co-occurring entities that are in relation r according to a knowledge base.
        #  For each relation, we rank the words by their tf-idf to learn, for example, that born is a good indicator for the relation place of birth."
        recognized_relations = set()
        for token in self.doc:
            if token.pos_ == "VERB" or token.pos_ == "NOUN":
                if token.lemma_ in top_relations:
                    recognized_relations.add(token.lemma_)
                else:
                    relation, score = self.find_most_similar_relation(token.lemma_, nlp, top_relations)
                    if score > 0.4:
                        recognized_relations.add(relation)

        recognized_relations = {f"<https://deepset.ai/harry_potter/{recognized_relation.replace(' ', '_')}>" for recognized_relation in recognized_relations}
        return recognized_relations

    def find_most_similar_relation(self, word, nlp, top_relations):
        relations_with_scores = [(relation, self.word_similarity(word, relation, nlp)) for relation in top_relations]
        relations_with_scores.sort(key=itemgetter(1), reverse=True)
        return relations_with_scores[0]

    def word_similarity(self, word1, word2, nlp):
        tokens = nlp(word1 + " " + word2)
        # for token in tokens:
        #    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
        token1, token2 = tokens[0], tokens[1]
        if token1.is_oov or token2.is_oov:
            return -1
        return token1.similarity(token2)
