import logging
from typing import List, Union, Optional, Iterator
import itertools
import torch

from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

from haystack.errors import HaystackError
from haystack.schema import Document
from haystack.nodes.base import BaseComponent
from haystack.nodes.preprocessor import PreProcessor
from haystack.modeling.utils import initialize_device_settings

logger = logging.getLogger(__name__)


class QuestionGenerator(BaseComponent):
    """
    The QuestionGenerator takes only a document as input and outputs questions that it thinks this document can answer. In the current implementation, it splits input texts into chunks of 50 words
    with a 10 word overlap. This is because the default model `valhalla/t5-base-e2e-qg` seems to generate only
    about 3 questions per passage, regardless of length.

    Our approach prioritizes the creation of more questions
    over processing efficiency (T5 can digest much more than 50 words at once). The returned questions
    generally come in an order dictated by the order of their answers, this means early questions in the list generally
    come from earlier in the document.
    """

    outgoing_edges = 1

    def __init__(
        self,
        model_name_or_path: str = "valhalla/t5-base-e2e-qg",
        model_version: Optional[str] = None,
        num_beams: int = 4,
        max_length: int = 256,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 1.5,
        early_stopping: bool = True,
        split_length: int = 50,
        split_overlap: int = 10,
        use_gpu: bool = True,
        prompt: str = "generate questions:",
        num_queries_per_doc: int = 1,
        sep_token: str = "<sep>",
        batch_size: int = 16,
        progress_bar: bool = True,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, torch.device]]] = None,
    ):
        """
        Uses the valhalla/t5-base-e2e-qg model by default. This class supports any question generation model that is implemented as a Seq2SeqLM in Hugging Face Transformers.
        Note that this style of question generation (where the only input
        is a document) is sometimes referred to as end-to-end question generation. Answer-supervised question
        generation is not currently supported.

        :param model_name_or_path: Directory of a saved model or the name of a public model, for example "valhalla/t5-base-e2e-qg".
                                   See [Hugging Face models](https://huggingface.co/models) for a full list of available models.
        :param model_version: The version of the model to use from the Hugging Face model hub. Can be a tag name, a branch name, or a commit hash.
        :param num_beams: The number of beams for beam search. `1` means no beam search.
        :param max_length: The maximum number of characters the generated text can have.
        :param no_repeat_ngram_size: If set to a number larger than 0, all ngrams whose size equals this number can only occur once. For example, if you set it to `3`, all 3-grams can appear once.
        :param length_penalty: Encourages the model to generate longer or shorter texts, depending on the value you specify. Values greater than 0.0 promote longer sequences, while values less than 0.0 promote shorter sequences. Used with text generation based on beams.
        :param early_stopping: Defines the stopping condition for beam search.
                                `True` means the model stops generating text after reaching the `num_beams`.
                                `False` means the model stops generating text only if it's unlikely to find better candidates.
        :param split_length: Determines the length of the split (a chunk of a document). Used by `num_queries_per_doc`.
        :param split_overlap: Configures the amount of overlap between two adjacent documents after a split. Setting it to a positive number enables sliding window approach.
        :param use_gpu: Whether to use GPU or the CPU. Falls back on CPU if no GPU is available.
        :param prompt: Contains the prompt with instructions for the model.
        :param batch_size: Number of documents to process at a time.
        :param num_queries_per_doc: Number of questions to generate per document. However, this is actually a number
                                    of questions to generate per split in the document where the `split_length` determines
                                    the length of the split and the `split_overlap` determines the overlap between splits.
                                    Therefore, this parameter is multiplied by the resulting number of splits to get the
                                    total number of questions generated per document. This value is capped at 3.
        :param sep_token: A special token that separates two sentences in the same output.
        :param progress_bar: Whether to show a tqdm progress bar or not.
        :param use_auth_token: The API token used to download private models from Hugging Face.
                               If set to `True`, the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) is used.
                               For more information, see [Hugging Face](https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained).
        :param devices: List of torch devices (for example cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects or strings is supported (for example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). If you specify `use_gpu=False`, the devices
                        parameter is not used and a single CPU device is used for inference.

        """
        super().__init__()
        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)
        if len(self.devices) > 1:
            logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                self.devices[0],
            )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path, revision=model_version, use_auth_token=use_auth_token
        )
        self.model.to(str(self.devices[0]))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token=use_auth_token)
        self.num_beams = num_beams
        self.max_length = max_length
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.preprocessor = PreProcessor()
        self.prompt = prompt
        self.num_queries_per_doc = min(num_queries_per_doc, 3)
        self.batch_size = batch_size
        self.sep_token = self.tokenizer.sep_token or sep_token
        self.progress_bar = progress_bar

    def run(self, documents: List[Document]):  # type: ignore
        generated_questions = []
        for d in documents:
            questions = self.generate(d.content)
            curr_dict = {"document_id": d.id, "document_sample": d.content[:200], "questions": questions}
            generated_questions.append(curr_dict)
        output = {"generated_questions": generated_questions, "documents": documents}
        return output, "output_1"

    def run_batch(self, documents: Union[List[Document], List[List[Document]]], batch_size: Optional[int] = None):  # type: ignore
        generated_questions = []
        if isinstance(documents[0], Document):
            questions = self.generate_batch(
                texts=[d.content for d in documents if isinstance(d, Document)], batch_size=batch_size
            )
            questions_iterator = questions  # type: ignore
            documents_iterator = documents
        else:
            questions = self.generate_batch(
                texts=[[d.content for d in doc_list] for doc_list in documents if isinstance(doc_list, list)],
                batch_size=batch_size,
            )
            questions_iterator = itertools.chain.from_iterable(questions)  # type: ignore
            documents_iterator = itertools.chain.from_iterable(documents)  # type: ignore
        for cur_questions, doc in zip(questions_iterator, documents_iterator):
            if not isinstance(doc, Document):
                raise HaystackError(f"doc was of type {type(doc)}, but expected a Document.")
            curr_dict = {"document_id": doc.id, "document_sample": doc.content[:200], "questions": cur_questions}
            generated_questions.append(curr_dict)
        output = {"generated_questions": generated_questions, "documents": documents}
        return output, "output_1"

    def generate(self, text: str) -> List[str]:
        # Performing splitting because T5 has a max input length
        # Also currently, it seems that it only generates about 3 questions for the beginning section of text
        split_texts_docs = self.preprocessor.split(
            document={"content": text},
            split_by="word",
            split_respect_sentence_boundary=False,
            split_overlap=self.split_overlap,
            split_length=self.split_length,
        )
        split_texts = [
            f"{self.prompt} {text.content}" if self.prompt not in text.content else text.content
            for text in split_texts_docs
        ]
        tokenized = self.tokenizer(split_texts, return_tensors="pt", padding=True)
        input_ids = tokenized["input_ids"].to(self.devices[0])
        # Necessary if padding is enabled so the model won't attend pad tokens
        attention_mask = tokenized["attention_mask"].to(self.devices[0])
        tokens_output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=self.num_beams,
            max_length=self.max_length,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            length_penalty=self.length_penalty,
            early_stopping=self.early_stopping,
            num_return_sequences=self.num_queries_per_doc,
        )

        string_output = self.tokenizer.batch_decode(tokens_output, skip_special_tokens=True)

        ret = []
        for split in string_output:
            for question in split.split(self.sep_token):
                question = question.strip()
                if question and question not in ret:
                    ret.append(question)

        return ret

    def generate_batch(
        self, texts: Union[List[str], List[List[str]]], batch_size: Optional[int] = None
    ) -> Union[List[List[str]], List[List[List[str]]]]:
        """
        Generates questions for a list of strings or a list of lists of strings.

        :param texts: List of str or list of list of str.
        :param batch_size: Number of texts to process at a time.
        """

        if batch_size is None:
            batch_size = self.batch_size

        if isinstance(texts[0], str):
            single_doc_list = True
            number_of_docs = [1 for text_list in texts]
            text_iterator = texts
        else:
            single_doc_list = False
            number_of_docs = [len(text_list) for text_list in texts]
            text_iterator = itertools.chain.from_iterable(texts)  # type: ignore

        split_texts_docs = [
            self.preprocessor.split(
                document={"content": text},
                split_by="word",
                split_respect_sentence_boundary=False,
                split_overlap=self.split_overlap,
                split_length=self.split_length,
            )
            for text in text_iterator
        ]
        split_texts = [[doc.content for doc in split if isinstance(doc.content, str)] for split in split_texts_docs]
        number_of_splits = [len(split) for split in split_texts]
        flat_split_texts = [
            f"{self.prompt} {text}" if self.prompt not in text else text
            for text in itertools.chain.from_iterable(split_texts)
        ]

        batches = self._get_batches(flat_split_texts, batch_size=batch_size)
        all_string_outputs = []
        pb = tqdm(total=len(flat_split_texts), disable=not self.progress_bar, desc="Generating questions")
        for batch in batches:
            tokenized = self.tokenizer(batch, return_tensors="pt", padding=True)
            input_ids = tokenized["input_ids"].to(self.devices[0])
            # Necessary if padding is enabled so the model won't attend pad tokens
            attention_mask = tokenized["attention_mask"].to(self.devices[0])
            tokens_output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=self.num_beams,
                max_length=self.max_length,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                length_penalty=self.length_penalty,
                early_stopping=self.early_stopping,
                num_return_sequences=self.num_queries_per_doc,
            )

            string_output = self.tokenizer.batch_decode(tokens_output, skip_special_tokens=True)
            all_string_outputs.extend(string_output)
            pb.update(len(batch))
        pb.close()
        # Group predictions together by split
        grouped_predictions_split = []
        left_idx = 0
        right_idx = 0
        for number in number_of_splits:
            right_idx = left_idx + number * self.num_queries_per_doc
            grouped_predictions_split.append(all_string_outputs[left_idx:right_idx])
            left_idx = right_idx
        # Group predictions together by doc list
        grouped_predictions_doc_list = []
        left_idx = 0
        right_idx = 0
        for number in number_of_docs:
            right_idx = left_idx + number
            grouped_predictions_doc_list.append(grouped_predictions_split[left_idx:right_idx])
            left_idx = right_idx

        results = []
        for group in grouped_predictions_doc_list:
            group_preds = []
            for doc in group:
                doc_preds = []
                for split in doc:
                    for question in split.split(self.sep_token):
                        question = question.strip()
                        if question and question not in doc_preds:
                            doc_preds.append(question)
                group_preds.append(doc_preds)
            if single_doc_list:
                results.append(group_preds[0])
            else:
                results.append(group_preds)

        return results

    @staticmethod
    def _get_batches(texts: List[str], batch_size: Optional[int]) -> Iterator[List[str]]:
        if batch_size is None:
            yield texts
            return
        else:
            for index in range(0, len(texts), batch_size):
                yield texts[index : index + batch_size]
