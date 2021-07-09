from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from haystack import BaseComponent

text = "generate questions: Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum \
and first released in 1991, Python's design philosophy emphasizes code \
readability with its notable use of significant whitespace."


"""
TODO
top_k instead of length
device and use gpu
Why does __init__ not work?
split to base class?
Not totally sure about args
"""

class QuestionGenerator(BaseComponent):
    def __init__(self,
                 model_name_or_path="valhalla/t5-base-e2e-qg",
                 model_version=None,
                 num_beams=4,
                 max_length=256,
                 no_repeat_ngram_size=3,
                 length_penalty=1.5,
                 early_stopping=True):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.set_config(
            model_name_or_path=model_name_or_path, model_version=model_version,
            max_length=max_length, num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty, early_stopping=early_stopping
        )
        self.num_beams = num_beams
        self.max_length = max_length
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping

    def generate(self, text):
        tokenized = self.tokenizer([text], return_tensors="pt")
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]   # necessary if padding is enabled so the model won't attend pad tokens
        tokens_output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=self.num_beams,
            max_length=self.max_length,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            length_penalty=self.length_penalty,
            early_stopping=self.early_stopping,
        )

        ret = [self.tokenizer.decode(t) for t in tokens_output]
        return ret

