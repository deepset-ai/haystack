from typing import List

from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

from haystack.schema import Document
from haystack.nodes.base import BaseComponent
from haystack.nodes.preprocessor import PreProcessor
from haystack.modeling.utils import initialize_device_settings


class QuestionGenerator(BaseComponent):
    """ 
    The Question Generator takes only a document as input and outputs questions that it thinks can be
    answered by this document. In our current implementation, input texts are split into chunks of 50 words
    with a 10 word overlap. This is because the default model `valhalla/t5-base-e2e-qg` seems to generate only
    about 3 questions per passage regardless of length. Our approach prioritizes the creation of more questions
    over processing efficiency (T5 is able to digest much more than 50 words at once). The returned questions
    generally come in an order dictated by the order of their answers i.e. early questions in the list generally
    come from earlier in the document.
    """
    outgoing_edges = 1

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
                 use_gpu=True,
                 prompt="generate questions:",
    ):
        """
        Uses the valhalla/t5-base-e2e-qg model by default. This class supports any question generation model that is
        implemented as a Seq2SeqLM in HuggingFace Transformers. Note that this style of question generation (where the only input
        is a document) is sometimes referred to as end-to-end question generation. Answer-supervised question
        generation is not currently supported.

        :param model_name_or_path: Directory of a saved model or the name of a public model e.g. "valhalla/t5-base-e2e-qg".
                                   See https://huggingface.co/models for full list of available models.
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param use_gpu: Whether to use GPU or the CPU. Falls back on CPU if no GPU is available.
        """
        self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.model.to(str(self.devices[0]))
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

    def run(self, documents: List[Document]):  # type: ignore
        generated_questions = []
        for d in documents:
            questions = self.generate(d.content)
            curr_dict = {"document_id": d.id,
                         "document_sample": d.content[:200],
                         "questions": questions}
            generated_questions.append(curr_dict)
        output = {"generated_questions": generated_questions, "documents": documents}
        return output, "output_1"

    def generate(self, text):
        # Performing splitting because T5 has a max input length
        # Also currently, it seems that it only generates about 3 questions for the beginning section of text
        split_texts_dict = self.preprocessor.split(
            document={"content": text},
            split_by="word",
            split_respect_sentence_boundary=False,
            split_overlap=self.split_overlap,
            split_length=self.split_length
        )
        split_texts = [x["content"] for x in split_texts_dict]
        ret = []
        for split_text in split_texts:
            if self.prompt not in split_text:
                split_text = self.prompt + " " + split_text
            tokenized = self.tokenizer([split_text], return_tensors="pt")
            input_ids = tokenized["input_ids"].to(self.devices[0])
            attention_mask = tokenized["attention_mask"].to(self.devices[0])   # necessary if padding is enabled so the model won't attend pad tokens
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
