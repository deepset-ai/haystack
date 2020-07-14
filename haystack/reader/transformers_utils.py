# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import numpy as np
from transformers.data import squad_convert_examples_to_features

logger = logging.getLogger(__name__)

from transformers.pipelines import pipeline
from transformers.pipelines import QuestionAnsweringPipeline as UpStreamQuestionAnsweringPipeline


# Adjusted the original class slightly to cope for https://github.com/huggingface/transformers/issues/5711
# until it is fixed upstream.
class QuestionAnsweringPipeline(UpStreamQuestionAnsweringPipeline):
    """
    Question Answering pipeline using ModelForQuestionAnswering head. See the
    `question answering usage <../usage.html#question-answering>`__ examples for more information.

    This question answering can currently be loaded from the :func:`~transformers.pipeline` method using
    the following task identifier(s):

    - "question-answering", for answering questions given a context.

    The models that this pipeline can use are models that have been fine-tuned on a question answering task.
    See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?filter=question-answering>`__.

    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`, defaults to :obj:`None`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`, defaults to :obj:`None`):
            The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to PyTorch.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`, defaults to :obj:`None`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to :obj:`-1`):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model
            on the associated CUDA device id.
    """

    default_input_names = "question,context"

    def __call__(self, *args, **kwargs):
        """
        Args:
            We support multiple use-cases, the following are exclusive:
            X: sequence of SquadExample
            data: sequence of SquadExample
            question: (str, List[str]), batch of question(s) to map along with context
            context: (str, List[str]), batch of context(s) associated with the provided question keyword argument
        Returns:
            dict: {'answer': str, 'score": float, 'start": int, "end": int}
            answer: the textual answer in the intial context
            score: the score the current answer scored for the model
            start: the character index in the original string corresponding to the beginning of the answer' span
            end: the character index in the original string corresponding to the ending of the answer' span
        """
        # Set defaults values
        kwargs.setdefault("topk", 1)
        kwargs.setdefault("doc_stride", 128)
        kwargs.setdefault("max_answer_len", 15)
        kwargs.setdefault("max_seq_len", 384)
        kwargs.setdefault("max_question_len", 64)
        kwargs.setdefault("handle_impossible_answer", False)

        if kwargs["topk"] < 1:
            raise ValueError("topk parameter should be >= 1 (got {})".format(kwargs["topk"]))

        if kwargs["max_answer_len"] < 1:
            raise ValueError("max_answer_len parameter should be >= 1 (got {})".format(kwargs["max_answer_len"]))

        # Convert inputs to features
        examples = self._args_parser(*args, **kwargs)
        features_list = [
            squad_convert_examples_to_features(
                examples=[example],
                tokenizer=self.tokenizer,
                max_seq_length=kwargs["max_seq_len"],
                doc_stride=kwargs["doc_stride"],
                max_query_length=kwargs["max_question_len"],
                is_training=False,
                tqdm_enabled=False,
            )
            for example in examples
        ]
        all_answers = []
        for features, example in zip(features_list, examples):
            model_input_names = self.tokenizer.model_input_names + ["input_ids"]
            fw_args = {k: [feature.__dict__[k] for feature in features] for k in model_input_names}

            # Manage tensor allocation on correct device
            with self.device_placement():
                if self.framework == "tf":
                    fw_args = {k: tf.constant(v) for (k, v) in fw_args.items()}
                    start, end = self.model(fw_args)[:2]
                    start, end = start.numpy(), end.numpy()
                else:
                    with torch.no_grad():
                        # Retrieve the score for the context tokens only (removing question tokens)
                        fw_args = {k: torch.tensor(v, device=self.device) for (k, v) in fw_args.items()}
                        start, end = self.model(**fw_args)[:2]
                        start, end = start.cpu().numpy(), end.cpu().numpy()

            min_null_score = 1000000  # large and positive
            answers = []
            for (feature, start_, end_) in zip(features, start, end):
                # Mask padding and question
                start_, end_ = (
                    start_ * np.abs(np.array(feature.p_mask) - 1),
                    end_ * np.abs(np.array(feature.p_mask) - 1),
                )

                # Mask CLS
                start_[0] = end_[0] = 0

                # Normalize logits and spans to retrieve the answer
                start_ = np.exp(start_ - np.log(np.sum(np.exp(start_), axis=-1, keepdims=True)))
                end_ = np.exp(end_ - np.log(np.sum(np.exp(end_), axis=-1, keepdims=True)))

                if kwargs["handle_impossible_answer"]:
                    min_null_score = min(min_null_score, (start_[0] * end_[0]).item())

                starts, ends, scores = self.decode(start_, end_, kwargs["topk"], kwargs["max_answer_len"])
                char_to_word = np.array(example.char_to_word_offset)

                # Convert the answer (tokens) back to the original text
                for s, e, score in zip(starts, ends, scores):
                    # CUSTOM ADJUSTMENT: We added this try-catch block here to not crash when model predicts
                    # start / end to be the final [SEP] token (https://github.com/huggingface/transformers/issues/5711)
                    try:
                        answers += [
                            {
                                "score": score.item(),
                                "start": np.where(char_to_word == feature.token_to_orig_map[s])[0][0].item(),
                                "end": np.where(char_to_word == feature.token_to_orig_map[e])[0][-1].item(),
                                "answer": " ".join(
                                    example.doc_tokens[feature.token_to_orig_map[s]: feature.token_to_orig_map[e] + 1]
                                ),
                            }]
                    except KeyError as e:
                        logger.warning(
                            f"Could not map predicted span ({s},{e}) back to token space. Skipping this prediction ...")

            if kwargs["handle_impossible_answer"]:
                answers.append({"score": min_null_score, "start": 0, "end": 0, "answer": ""})

            answers = sorted(answers, key=lambda x: x["score"], reverse=True)[: kwargs["topk"]]
            all_answers += answers

        if len(all_answers) == 1:
            return all_answers[0]
        return all_answers
