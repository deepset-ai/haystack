# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.components.evaluators import (
    AnswerExactMatchEvaluator,
    DocumentMAPEvaluator,
    DocumentMRREvaluator,
    DocumentRecallEvaluator,
)


@pytest.mark.parametrize(
    "evaluator, kwargs",
    [
        (AnswerExactMatchEvaluator(), {"ground_truth_answers": [], "predicted_answers": []}),
        (DocumentMAPEvaluator(), {"ground_truth_documents": [], "retrieved_documents": []}),
        (DocumentMRREvaluator(), {"ground_truth_documents": [], "retrieved_documents": []}),
        (DocumentRecallEvaluator(), {"ground_truth_documents": [], "retrieved_documents": []}),
    ],
)
def test_run_with_empty_inputs_raises_value_error(evaluator, kwargs):
    # Empty (equal-length) inputs previously fell through the length check and
    # crashed with a bare ZeroDivisionError when averaging over zero items.
    # They must instead raise a descriptive ValueError, matching
    # DocumentNDCGEvaluator's contract.
    with pytest.raises(ValueError, match="must be provided"):
        evaluator.run(**kwargs)
