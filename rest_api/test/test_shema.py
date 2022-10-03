import unittest

from rest_api.rest_api.schema import QuestionAnswerPair


class TestObjectModels(unittest.TestCase):
    def test_question_answer_pair(self):
        qap = QuestionAnswerPair(question="a question", answer="an answer", game="monopoly", approved=False)
        qap_dict = qap.dict()
        self.assertDictEqual(qap_dict, dict(question="a question", answer="an answer", game="monopoly", approved=False))
        qap_from_dict = QuestionAnswerPair(**qap_dict)
        self.assertEqual(qap, qap_from_dict)
