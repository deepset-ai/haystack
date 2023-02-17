from typing import List, Tuple, Optional, Union
import logging
from transformers import AutoConfig
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)


def semantic_answer_similarity(
    predictions: List[List[str]],
    gold_labels: List[List[str]],
    sas_model_name_or_path: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    batch_size: int = 32,
    use_gpu: bool = True,
    use_auth_token: Optional[Union[str, bool]] = None,
) -> Tuple[List[float], List[float], List[List[float]]]:
    """
    Computes Transformer-based similarity of predicted answer to gold labels to derive a more meaningful metric than EM or F1.
    Returns per QA pair a) the similarity of the most likely prediction (top 1) to all available gold labels
                        b) the highest similarity of all predictions to gold labels
                        c) a matrix consisting of the similarities of all the predictions compared to all gold labels

    :param predictions: Predicted answers as list of multiple preds per question
    :param gold_labels: Labels as list of multiple possible answers per question
    :param sas_model_name_or_path: SentenceTransformers semantic textual similarity model, should be path or string
                                     pointing to downloadable models.
    :param batch_size: Number of prediction label pairs to encode at once.
    :param use_gpu: Whether to use a GPU or the CPU for calculating semantic answer similarity.
                    Falls back to CPU if no GPU is available.
    :param use_auth_token: The API token used to download private models from Huggingface.
                           If this parameter is set to `True`, then the token generated when running
                           `transformers-cli login` (stored in ~/.huggingface) will be used.
                           Additional information can be found here
                           https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
    :return: top_1_sas, top_k_sas, pred_label_matrix
    """
    assert len(predictions) == len(gold_labels)

    config = AutoConfig.from_pretrained(sas_model_name_or_path, use_auth_token=use_auth_token)
    cross_encoder_used = False
    if config.architectures is not None:
        cross_encoder_used = any(arch.endswith("ForSequenceClassification") for arch in config.architectures)

    device = None if use_gpu else "cpu"

    # Compute similarities
    top_1_sas = []
    top_k_sas = []
    pred_label_matrix = []
    lengths: List[Tuple[int, int]] = []

    # Based on Modelstring we can load either Bi-Encoders or Cross Encoders.
    # Similarity computation changes for both approaches
    if cross_encoder_used:
        model = CrossEncoder(
            sas_model_name_or_path,
            device=device,
            tokenizer_args={"use_auth_token": use_auth_token},
            automodel_args={"use_auth_token": use_auth_token},
        )
        grid = []
        for preds, labels in zip(predictions, gold_labels):
            for p in preds:
                for l in labels:
                    grid.append((p, l))
            lengths.append((len(preds), len(labels)))
        scores = model.predict(grid, batch_size=batch_size)

        current_position = 0
        for len_p, len_l in lengths:
            scores_window = scores[current_position : current_position + len_p * len_l]
            # Per predicted doc there are len_l entries comparing it to all len_l labels.
            # So to only consider the first doc we have to take the first len_l entries
            top_1_sas.append(np.max(scores_window[:len_l]))
            top_k_sas.append(np.max(scores_window))
            pred_label_matrix.append(scores_window.reshape(len_p, len_l).tolist())
            current_position += len_p * len_l
    else:
        # For Bi-encoders we can flatten predictions and labels into one list
        model = SentenceTransformer(sas_model_name_or_path, device=device, use_auth_token=use_auth_token)
        all_texts: List[str] = []
        for p, l in zip(predictions, gold_labels):  # type: ignore
            # TODO potentially exclude (near) exact matches from computations
            all_texts.extend(p)
            all_texts.extend(l)
            lengths.append((len(p), len(l)))
        # then compute embeddings
        embeddings = model.encode(all_texts, batch_size=batch_size)

        # then select which embeddings will be used for similarity computations
        current_position = 0
        for len_p, len_l in lengths:
            pred_embeddings = embeddings[current_position : current_position + len_p, :]
            current_position += len_p
            label_embeddings = embeddings[current_position : current_position + len_l, :]
            current_position += len_l
            sims = cosine_similarity(pred_embeddings, label_embeddings)
            top_1_sas.append(np.max(sims[0, :]))
            top_k_sas.append(np.max(sims))
            pred_label_matrix.append(sims.tolist())

    return top_1_sas, top_k_sas, pred_label_matrix
