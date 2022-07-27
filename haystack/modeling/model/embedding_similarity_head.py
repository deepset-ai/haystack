from typing import Tuple, Union, Optional

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import NLLLoss

from haystack.modeling.utils import all_gather_list
from haystack.modeling.model.prediction_head import PredictionHead


logger = logging.getLogger(__name__)


def dot_product_scores(query_vectors: torch.Tensor, passage_vectors: torch.Tensor) -> torch.Tensor:
    """
    Calculates dot product similarity scores for two 2-dimensional tensors

    :param query_vectors: tensor of query embeddings from BiAdaptive model
                    of dimension n1 x D,
                    where n1 is the number of queries/batch size and D is embedding size
    :param passage_vectors: tensor of context/passage embeddings from BiAdaptive model
                    of dimension n2 x D,
                    where n2 is (batch_size * num_positives) + (batch_size * num_hard_negatives)
                    and D is embedding size

    :return: dot_product: similarity score of each query with each context/passage (dimension: n1xn2)
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    dot_product = torch.matmul(query_vectors, torch.transpose(passage_vectors, 0, 1))
    return dot_product


def cosine_scores(query_vectors: torch.Tensor, passage_vectors: torch.Tensor) -> torch.Tensor:
    """
    Calculates cosine similarity scores for two 2-dimensional tensors

    :param query_vectors: tensor of query embeddings from BiAdaptive model
                        of dimension n1 x D,
                        where n1 is the number of queries/batch size and D is embedding size
    :param passage_vectors: tensor of context/passage embeddings from BiAdaptive model
                        of dimension n2 x D,
                        where n2 is (batch_size * num_positives) + (batch_size * num_hard_negatives)
                        and D is embedding size

    :return: cosine similarity score of each query with each context/passage (dimension: n1xn2)
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    cosine_similarities = []
    passages_per_batch = passage_vectors.shape[0]
    for query_vector in query_vectors:
        query_vector_repeated = query_vector.repeat(passages_per_batch, 1)
        current_cosine_similarities = nn.functional.cosine_similarity(query_vector_repeated, passage_vectors, dim=1)
        cosine_similarities.append(current_cosine_similarities)
    return torch.stack(cosine_similarities)




SIMILARITY_FUNCTIONS = {
    "dot_product": dot_product_scores,
    "cosine": cosine_scores,
}


# Based on TextSimilarityHead

class EmbeddingSimilarityHead(PredictionHead):
    """
    Trains a head on predicting the similarity of two embeddings.
    """

    def __init__(self, similarity_function: str = "dot_product", global_loss_buffer_size: int = 150000):
        """
        :param similarity_function: Function to calculate similarity between queries and passage embeddings.
                                    Choose either "dot_product" (Default) or "cosine".
        :param global_loss_buffer_size: Buffer size for all_gather() in DDP.
                                        Increase if errors like "encoded data exceeds max_size ..." come up
        """
        super().__init__()

        self.similarity_function = similarity_function
        self.loss_fct = NLLLoss(reduction="mean")
        self.task_name = "text_similarity"
        self.model_type = "text_similarity"
        self.ph_output_type = "per_sequence"
        self.global_loss_buffer_size = global_loss_buffer_size


    # def forward(self, query_vectors: torch.Tensor, passage_vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Only packs the embeddings from both language models into a tuple. No further modification.
    #     The similarity calculation is handled later to enable distributed training (DDP)
    #     while keeping the support for in-batch negatives.
    #     (Gather all embeddings from nodes => then do similarity scores + loss)

    #     :param query_vectors: Tensor of query embeddings from BiAdaptive model
    #                       of dimension n1 x D,
    #                       where n1 is the number of queries/batch size and D is embedding size
    #     :param passage_vectors: Tensor of context/passage embeddings from BiAdaptive model
    #                       of dimension n2 x D,
    #                       where n2 is the number of queries/batch size and D is embedding size
    #     """
    #     return query_vectors, passage_vectors

    def _embeddings_to_scores(self, query_vectors: torch.Tensor, passage_vectors: torch.Tensor) -> torch.Tensor:
        """
        Calculates similarity scores between all given query_vectors and passage_vectors

        :param query_vectors: Tensor of queries encoded by the query encoder model
        :param passage_vectors: Tensor of passages encoded by the passage encoder model
        :return: Tensor of log softmax similarity scores of each query with each passage (dimension: n1xn2)
        """
        try:
            sim_func = SIMILARITY_FUNCTIONS[self.similarity_function]
        except KeyError:
            raise AttributeError(
                f"The similarity function can only be 'dot_product' or 'cosine', not '{self.similarity_function}'"
            )
        scores = sim_func(query_vectors, passage_vectors)

        if len(query_vectors.size()) > 1:
            q_num = query_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = nn.functional.log_softmax(scores, dim=1)
        return softmax_scores

    def logits_to_loss(self, logits: Tuple[torch.Tensor, torch.Tensor], label_ids, **kwargs):
        """
        Computes the loss (Default: NLLLoss) by applying a similarity function (Default: dot product) to the input
        tuple of (query_vectors, passage_vectors) and afterwards applying the loss function on similarity scores.

        :param logits: Tuple of Tensors (query_embedding, passage_embedding) as returned from forward()

        :return: negative log likelihood loss from similarity scores
        """
        # Check if DDP is initialized
        try:
            if torch.distributed.is_available():
                rank = torch.distributed.get_rank()
            else:
                rank = -1
        except (AssertionError, RuntimeError) as e:
            logging.debug(e)
            rank = -1

        # Prepare predicted scores
        query_vectors, passage_vectors = logits

        # Prepare Labels
        positive_idx_per_question = torch.nonzero((label_ids.view(-1) == 1), as_tuple=False)

        global_query_vectors = []
        global_passage_vectors = []
        global_positive_idx_per_question = []

        if rank == -1:
            global_query_vectors = query_vectors
            global_passage_vectors = passage_vectors
            global_positive_idx_per_question = positive_idx_per_question
        else:
            # Gather global embeddings from all distributed nodes (DDP)
            q_vector_to_send = torch.empty_like(query_vectors).cpu().copy_(query_vectors).detach_()
            p_vector_to_send = torch.empty_like(passage_vectors).cpu().copy_(passage_vectors).detach_()

            global_question_passage_vectors = all_gather_list(
                [q_vector_to_send, p_vector_to_send, positive_idx_per_question], max_size=self.global_loss_buffer_size
            )

            global_query_vectors = []
            global_passage_vectors = []
            global_positive_idx_per_question = []
            total_passages = 0
            for i, item in enumerate(global_question_passage_vectors):
                q_vector, p_vectors, positive_idx = item

                if i != rank:
                    global_query_vectors.append(q_vector.to(query_vectors.device))
                    global_passage_vectors.append(p_vectors.to(passage_vectors.device))
                    global_positive_idx_per_question.extend([v + total_passages for v in positive_idx])
                else:
                    global_query_vectors.append(query_vectors)
                    global_passage_vectors.append(passage_vectors)
                    global_positive_idx_per_question.extend([v + total_passages for v in positive_idx_per_question])
                total_passages += p_vectors.size(0)

            global_query_vectors = torch.cat(global_query_vectors, dim=0)
            global_passage_vectors = torch.cat(global_passage_vectors, dim=0)
            global_positive_idx_per_question = torch.LongTensor(global_positive_idx_per_question)


        # Get similarity scores
        softmax_scores = self._embeddings_to_scores(global_query_vectors, global_passage_vectors)
        targets = global_positive_idx_per_question.squeeze(-1).to(softmax_scores.device)

        # Calculate loss
        loss = self.loss_fct(softmax_scores, targets)
        return loss

    def logits_to_preds(self, logits: Tuple[torch.Tensor, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Returns predicted ranks(similarity) of passages/context for each query

        :param logits: tensor of log softmax similarity scores of each query with each context/passage (dimension: n1xn2)

        :return: predicted ranks of passages for each query
        """
        query_vectors, passage_vectors = logits
        softmax_scores = self._embeddings_to_scores(query_vectors, passage_vectors)
        _, sorted_scores = torch.sort(softmax_scores, dim=1, descending=True)
        return sorted_scores

    def prepare_labels(self, label_ids) -> torch.Tensor:
        """
        Returns a tensor with passage labels(0:hard_negative/1:positive) for each query

        :return: passage labels(0:hard_negative/1:positive) for each query
        """
        labels = torch.zeros(label_ids.size(0), label_ids.numel())

        positive_indices = torch.nonzero(label_ids.view(-1) == 1, as_tuple=False)

        for i, indx in enumerate(positive_indices):
            labels[i, indx.item()] = 1
        return labels

    def formatted_preds(self, logits: Tuple[torch.Tensor, torch.Tensor], **kwargs):
        raise NotImplementedError("formatted_preds is not supported in TextSimilarityHead yet!")
