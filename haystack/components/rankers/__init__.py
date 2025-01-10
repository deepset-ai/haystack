# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from haystack.lazy_imports import lazy_dir, lazy_getattr

if TYPE_CHECKING:
    from haystack.components.rankers.lost_in_the_middle import LostInTheMiddleRanker
    from haystack.components.rankers.meta_field import MetaFieldRanker
    from haystack.components.rankers.meta_field_grouping_ranker import MetaFieldGroupingRanker
    from haystack.components.rankers.sentence_transformers_diversity import SentenceTransformersDiversityRanker
    from haystack.components.rankers.transformers_similarity import TransformersSimilarityRanker


_lazy_imports = {
    "LostInTheMiddleRanker": "haystack.components.rankers.lost_in_the_middle",
    "MetaFieldRanker": "haystack.components.rankers.meta_field",
    "MetaFieldGroupingRanker": "haystack.components.rankers.meta_field_grouping_ranker",
    "SentenceTransformersDiversityRanker": "haystack.components.rankers.sentence_transformers_diversity",
    "TransformersSimilarityRanker": "haystack.components.rankers.transformers_similarity",
}

__all__ = list(_lazy_imports.keys())


def __getattr__(name):
    return lazy_getattr(name, _lazy_imports, __name__)


def __dir__():
    return lazy_dir(_lazy_imports)
