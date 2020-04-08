from transformers import pipeline


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
        context_size=30,
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
        :param context_size: num of chars (before and after the answer) to return as "context" for each answer.
                            The context usually helps users to understand if the answer really makes sense.
        :param use_gpu: < 1  -> use cpu
                        >= 1 -> num of gpus to use
        """
        self.model = pipeline("question-answering", model=model, tokenizer=tokenizer, device=use_gpu)
        self.context_size = context_size
        self.n_best_per_passage = n_best_per_passage
        #TODO param to modify bias for no_answer


    def predict(self, question, paragraphs, meta_data_paragraphs=None, top_k=None):
        """
        Use loaded QA model to find answers for a question in the supplied paragraphs.

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
        :param paragraphs: list of strings in which to search for the answer
        :param meta_data_paragraphs: list of dicts containing meta data for the paragraphs.
                                     len(paragraphs) == len(meta_data_paragraphs)
        :param top_k: the maximum number of answers to return
        :param max_processes: max number of parallel processes
        :return: dict containing question and answers

        """
        #TODO pass metadata

        # get top-answers for each candidate passage
        answers = []
        for i,p in enumerate(paragraphs):
            query = {"context": p, "question": question}
            predictions = self.model(query, topk=self.n_best_per_passage)
            # assemble and format all answers
            for pred in predictions:
                if pred["answer"]:
                    context_start = max(0, pred["start"] - self.context_size)
                    context_end = min(len(p), pred["end"] + self.context_size)
                    answers.append({
                        "answer": pred["answer"],
                        "context": p[context_start:context_end],
                        "offset_answer_start": pred["start"],
                        "offset_answer_end": pred["end"],
                        "probability": pred["score"],  
                        "score": None,
                        "document_id": meta_data_paragraphs[i]["document_id"],
                        "document_name":meta_data_paragraphs[i]["document_name"]
                    })

        # sort answers by their `probability` and select top-k
        answers = sorted(
            answers, key=lambda k: k["probability"], reverse=True
        )
        answers = answers[:top_k]

        results = {"question": question,
                   "answers": answers}

        return results
