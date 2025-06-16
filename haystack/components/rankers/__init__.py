# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "hugging_face_tei": ["HuggingFaceTEIRanker"],
    "lost_in_the_middle": ["LostInTheMiddleRanker"],
    "meta_field": ["MetaFieldRanker"],
    "meta_field_grouping_ranker": ["MetaFieldGroupingRanker"],
    "sentence_transformers_diversity": ["SentenceTransformersDiversityRanker"],
    "sentence_transformers_similarity": ["SentenceTransformersSimilarityRanker"],
    "transformers_similarity": ["TransformersSimilarityRanker"],
}

if TYPE_CHECKING:
    from .hugging_face_tei import HuggingFaceTEIRanker as HuggingFaceTEIRanker
    from .lost_in_the_middle import LostInTheMiddleRanker as LostInTheMiddleRanker
    from .meta_field import MetaFieldRanker as MetaFieldRanker
    from .meta_field_grouping_ranker import MetaFieldGroupingRanker as MetaFieldGroupingRanker
    from .sentence_transformers_diversity import (
        SentenceTransformersDiversityRanker as SentenceTransformersDiversityRanker,
    )
    from .sentence_transformers_similarity import (
        SentenceTransformersSimilarityRanker as SentenceTransformersSimilarityRanker,
    )
    from .transformers_similarity import TransformersSimilarityRanker as TransformersSimilarityRanker

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
