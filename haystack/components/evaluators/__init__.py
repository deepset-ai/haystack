# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "answer_exact_match": ["AnswerExactMatchEvaluator"],
    "context_relevance": ["ContextRelevanceEvaluator"],
    "document_map": ["DocumentMAPEvaluator"],
    "document_mrr": ["DocumentMRREvaluator"],
    "document_ndcg": ["DocumentNDCGEvaluator"],
    "document_recall": ["DocumentRecallEvaluator"],
    "faithfulness": ["FaithfulnessEvaluator"],
    "llm_evaluator": ["LLMEvaluator"],
    "sas_evaluator": ["SASEvaluator"],
}

if TYPE_CHECKING:
    from .answer_exact_match import AnswerExactMatchEvaluator as AnswerExactMatchEvaluator
    from .context_relevance import ContextRelevanceEvaluator as ContextRelevanceEvaluator
    from .document_map import DocumentMAPEvaluator as DocumentMAPEvaluator
    from .document_mrr import DocumentMRREvaluator as DocumentMRREvaluator
    from .document_ndcg import DocumentNDCGEvaluator as DocumentNDCGEvaluator
    from .document_recall import DocumentRecallEvaluator as DocumentRecallEvaluator
    from .faithfulness import FaithfulnessEvaluator as FaithfulnessEvaluator
    from .llm_evaluator import LLMEvaluator as LLMEvaluator
    from .sas_evaluator import SASEvaluator as SASEvaluator

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
