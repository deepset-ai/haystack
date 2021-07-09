from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from haystack import BaseComponent
from haystack.preprocessor import PreProcessor

text = "generate questions: Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum \
and first released in 1991, Python's design philosophy emphasizes code \
readability with its notable use of significant whitespace."


"""
TODO
top_k instead of length?
device and use gpu
split to base class?
Not totally sure about args
run method
Does it clip long texts??
https://discuss.huggingface.co/t/t5-finetuning-tips/684
Inefficient because we feed it 50 words
"""

class QuestionGenerator(BaseComponent):
    def __init__(self,
                 model_name_or_path="valhalla/t5-base-e2e-qg",
                 model_version=None,
                 num_beams=4,
                 max_length=256,
                 no_repeat_ngram_size=3,
                 length_penalty=1.5,
                 early_stopping=True,
                 split_length=50,
                 split_overlap=10,
                 prompt="generate questions:"):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.set_config(
            model_name_or_path=model_name_or_path, model_version=model_version,
            max_length=max_length, num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty, early_stopping=early_stopping, split_length=split_length,
            split_overlap=split_overlap
        )
        self.num_beams = num_beams
        self.max_length = max_length
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.preprocessor = PreProcessor()
        self.prompt = prompt

    def generate(self, text):
        # Performing splitting because T5 has a max input length
        # Also currently, it seems that it only generates about 3 questions for the beginning section of text
        split_texts_dict = self.preprocessor.split(
            document={"text": text},
            split_by="word",
            split_respect_sentence_boundary=False,
            split_overlap=self.split_overlap,
            split_length=self.split_length
        )
        split_texts = [x["text"] for x in split_texts_dict]
        ret = []
        for split_text in split_texts:
            if self.prompt not in split_text:
                split_text = self.prompt + " " + split_text
            tokenized = self.tokenizer([split_text], return_tensors="pt")
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

            string_output = self.tokenizer.decode(tokens_output[0])
            string_output = string_output.replace("<pad>", "").replace("</s>", "")
            questions_string = string_output.split("<sep>")
            questions = [x for x in questions_string if x]

            # Doing this instead of set to maintain order since the generated questions seem to have answers
            # that occur in order in the text
            for q in questions:
                if q not in ret:
                    ret.append(q)
        return ret

