import json
import logging
import urllib.parse
from typing import Optional, Set
from spacy.tokens import Doc
from operator import itemgetter
from enum import Enum

logger = logging.getLogger(__name__)


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

    def analyze(self, nlp, alias_to_entity_and_prob, subject_names, predicate_names, object_names, relation_tfidf) -> (
    Optional[Set[str]], Optional[Set[str]], Optional[QuestionType]):
        self.doc = nlp(self.question_text)
        self.entities = self.entity_linking(alias_to_entity_and_prob=alias_to_entity_and_prob,
                                            subject_names=subject_names,
                                            object_names=object_names)
        self.relations = self.relation_linking(predicate_names=predicate_names, nlp=nlp, relation_tfidf=relation_tfidf)
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

    def entity_linking(self, alias_to_entity_and_prob, subject_names, object_names):
        """
        Link spacy entities mentioned in a question to entities that exist in our knowledge base
        """
        linked_entities_indices = set()
        entities = set()

        for np in self.doc.noun_chunks:
            # remove leading determiners a, an, and the from noun chunk
            np_lemma_without_determiners = np.lemma_
            np_lemma_without_determiners = np_lemma_without_determiners.replace(" \'", "")
            if np.lemma_.startswith("a "):
                np_lemma_without_determiners = np_lemma_without_determiners[2:]
            if np.lemma_.startswith("an "):
                np_lemma_without_determiners = np_lemma_without_determiners[3:]
            if np.lemma_.startswith("the "):
                np_lemma_without_determiners = np_lemma_without_determiners[4:]

            # todo prevent who from being matched to Bartemius crouch junior
            if np_lemma_without_determiners == "who":
                continue

            if self.add_namespace_to_resource(np_lemma_without_determiners) in subject_names or self.add_namespace_to_resource(
                    np_lemma_without_determiners) in object_names:
                entities.add(np_lemma_without_determiners)
                linked_entities_indices.update(range(np.start, np.end))

            elif np_lemma_without_determiners in alias_to_entity_and_prob:
                entities.add(max(alias_to_entity_and_prob[np_lemma_without_determiners], key=itemgetter(1))[0])
                linked_entities_indices.update(range(np.start, np.end))

        if self.doc.ents:  # if spacy recognized any entities
            for ent in self.doc.ents:
                if ent.start in linked_entities_indices:
                    # skip tokens that have already been linked
                    continue
                if self.add_namespace_to_resource(ent.text) in subject_names or self.add_namespace_to_resource(
                        ent.text) in object_names:
                    entities.add(ent.text)
                    linked_entities_indices.update(range(ent.start, ent.end))
                elif ent.text in alias_to_entity_and_prob:
                    entities.add(max(alias_to_entity_and_prob[ent.text], key=itemgetter(1))[0])
                    linked_entities_indices.update(range(ent.start, ent.end))
        # for any remaining tokens: try to link nouns to entities
        for token in self.doc:
            if token.i in linked_entities_indices:
                # skip tokens that have already been linked
                continue
            if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                if self.add_namespace_to_resource(token.text) in subject_names or self.add_namespace_to_resource(
                        token.text) in object_names:
                    entities.add(token.text)
                elif token.text in alias_to_entity_and_prob:
                    entities.add(max(alias_to_entity_and_prob[token.text], key=itemgetter(1))[0])

            elif token.pos_ == "ADJ":
                if self.add_namespace_to_resource(token.text) in object_names:
                    entities.add(token.text)

        logger.info(f"linked entities: {entities}")
        return {self.add_namespace_to_resource(entity, brackets=True) for entity in entities}

    @staticmethod
    def add_namespace_to_resource(resource: str, brackets=False, capitalize=True) -> str:
        if capitalize:
            resource = resource.capitalize()
        else:
            resource = resource.lower()
        resource = resource.replace(' ', '_').replace('.', '_')
        # encode entity names with special characters, such as Ichirō Nagai
        resource = urllib.parse.quote_plus(resource)
        if brackets:
            resource = f"<https://deepset.ai/harry_potter/{resource}>"
        else:
            resource = f"https://deepset.ai/harry_potter/{resource}"
        return resource

    def relation_linking(self, predicate_names, nlp, relation_tfidf):
        """
        Link verbs and nouns mentioned in a question to relations that exist in our knowledge base
        """
        relations = set()
        for token in self.doc:
            if token.pos_ == "VERB" or token.pos_ == "NOUN" or token.pos_ == "PROPN":
                if self.add_namespace_to_resource(token.lemma_, brackets=False, capitalize=False) in predicate_names:
                    relations.add(token.lemma_)
                elif self.add_namespace_to_resource(token.text, brackets=False, capitalize=False) in predicate_names:
                    relations.add(token.text)
                elif token.pos_ == "VERB":
                    best_relation_result, highest_tfidf = self.best_relation(token.lemma_.lower(), predicate_names, relation_tfidf)
                    if best_relation_result and highest_tfidf > 800:
                        best_relation_result = best_relation_result.split("/")[-1]
                        relations.add(best_relation_result)
                    # fuzzy matching for relation names based on word vector similarity
                    #relation, score = self.find_most_similar_relation(token.lemma_, nlp, predicate_names)
                    #if score > 0.4:
                    #    logger.info(f"Adding relation {relation} for token {token.text}")
                    #    relations.add(relation)

        if not relations:
            if self.question_text.lower().startswith("when"):
                relations.add("date")
            if self.question_text.lower().startswith("where"):
                relations.add("location")
            for token in self.doc:
                if token.pos_ == "VERB" or token.pos_ == "NOUN" or token.pos_ == "PROPN":
                    best_relation_result, highest_tfidf = self.best_relation(token.lemma_.lower(), predicate_names, relation_tfidf)
                    if best_relation_result and highest_tfidf > 800:
                        best_relation_result = best_relation_result.split("/")[-1]
                        relations.add(best_relation_result)
                    relation, score = self.find_most_similar_relation(token.lemma_, nlp, predicate_names)
                    if score > 0.4:
                        logger.info(f"Adding relation {relation} for token {token.text}")
                        relations.add(relation)

        logger.info(f"linked relations: {relations}")
        relations = {self.add_namespace_to_resource(recognized_relation, brackets=True, capitalize=False) for recognized_relation in
                     relations}
        return relations

    def find_most_similar_relation(self, word: str, nlp, predicate_names):
        relations_with_scores = [(relation.split("/")[-1], self.word_similarity(word, relation.split("/")[-1], nlp)) for
                                 relation in predicate_names]
        relations_with_scores.sort(key=itemgetter(1), reverse=True)
        return relations_with_scores[0]

    def word_similarity(self, word1: str, word2: str, nlp):
        token1, token2 = nlp(word1)[0], nlp(word2)[0]
        if token1.is_oov or token2.is_oov:
            return -1
        return token1.similarity(token2)

    def lookup(self, token, relation, relation_tfidf):
        #relation = "https://deepset.ai/harry_potter/" + relation
        if str((token, relation)) in relation_tfidf:
            return relation_tfidf[str((token, relation))]
        else:
            return -1

    def best_relation(self, token, predicate_names, relation_tfidf):
        #token = self.nlp(token)[0].lemma_.lower()
        highest_tfidf = 0
        best_relation = None
        for relation in predicate_names:
            if relation.endswith("family") or relation.endswith("loyalty") or relation.endswith("alias") or relation.endswith("family_members") or relation.endswith("affiliation") or relation.endswith("wife") or relation.endswith("husband"):
                continue
            tfidf = self.lookup(token, relation, relation_tfidf)
            if tfidf > highest_tfidf:
                highest_tfidf = tfidf
                best_relation = relation
        return best_relation, highest_tfidf
