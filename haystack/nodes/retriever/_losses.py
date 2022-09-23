from collections import namedtuple
from typing import Dict

from sentence_transformers import losses


SentenceTransformerLoss = namedtuple("SentenceTransformerLoss", "loss required_attrs")

_TRAINING_LOSSES: Dict[str, SentenceTransformerLoss] = {
    "mnrl": SentenceTransformerLoss(losses.MultipleNegativesRankingLoss, {"question", "pos_doc"}),
    "margin_mse": SentenceTransformerLoss(losses.MarginMSELoss, {"question", "pos_doc", "neg_doc", "score"}),
}
