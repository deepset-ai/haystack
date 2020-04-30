from transformers import pipeline

from haystack.database.base import Document


class TransformersReader:
    """
    Transformer based model for extractive Question Answering using the huggingface's transformers framework
    (https://github.com/huggingface/transformers).
    While the underlying model can vary (BERT, Roberta, DistilBERT ...) the interface remains the same.

    With the reader, you can:
     - directly get predictions via predict()
    """

    def __init__(
        self,
        model="distilbert-base-uncased-distilled-squad",
        tokenizer="distilbert-base-uncased",
        context_window_size=30,
        #no_answer_shift=-100,
        #batch_size=16,
        use_gpu=0,
        n_best_per_passage=2
    ):
        """
        Load a QA model from Transformers.
        Available models include:
        - distilbert-base-uncased-distilled-squad
        - bert-large-cased-whole-word-masking-finetuned-squad
        - bert-large-uncased-whole-word-masking-finetuned-squad

        See https://huggingface.co/models for full list of available QA models

        :param model: name of the model
        :param tokenizer: name of the tokenizer (usually the same as model)
        :param context_window_size: num of chars (before and after the answer) to return as "context" for each answer.
                            The context usually helps users to understand if the answer really makes sense.
        :param use_gpu: < 1  -> use cpu
                        >= 0 -> ordinal of the gpu to use
        """
        self.model = pipeline("question-answering", model=model, tokenizer=tokenizer, device=use_gpu)
        self.context_window_size = context_window_size
        self.n_best_per_passage = n_best_per_passage
        #TODO param to modify bias for no_answer

    def predict(self, question: str, documents: [Document], top_k: int = None):
        """
        Use loaded QA model to find answers for a question in the supplied list of Document.

        Returns dictionaries containing answers sorted by (desc.) probability
        Example:
        {'question': 'Who is the father of Arya Stark?',
        'answers': [
                     {'answer': 'Eddard,',
                     'context': " She travels with her father, Eddard, to King's Landing when he is ",
                     'offset_answer_start': 147,
                     'offset_answer_end': 154,
                     'probability': 0.9787139466668613,
                     'score': None,
                     'document_id': None
                     },
                    ...
                   ]
        }

        :param question: question string
        :param documents: list of Document in which to search for the answer
        :param top_k: the maximum number of answers to return
        :return: dict containing question and answers

        """
        # get top-answers for each candidate passage
        answers = []
        for doc in documents:
            query = {"context": doc.text, "question": question}
            predictions = self.model(query, topk=self.n_best_per_passage)
            # assemble and format all answers
            for pred in predictions:
                if pred["answer"]:
                    context_start = max(0, pred["start"] - self.context_window_size)
                    context_end = min(len(doc.text), pred["end"] + self.context_window_size)
                    answers.append({
                        "answer": pred["answer"],
                        "context": doc.text[context_start:context_end],
                        "offset_answer_start": pred["start"],
                        "offset_answer_end": pred["end"],
                        "probability": pred["score"],
                        "score": None,
                        "document_id": doc.id
                    })

        # sort answers by their `probability` and select top-k
        answers = sorted(
            answers, key=lambda k: k["probability"], reverse=True
        )
        answers = answers[:top_k]

        results = {"question": question,
                   "answers": answers}

        return results
