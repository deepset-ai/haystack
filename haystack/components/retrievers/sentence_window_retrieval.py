# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import warnings

from .sentence_window_retriever import SentenceWindowRetriever


class SentenceWindowRetrieval(SentenceWindowRetriever):
    """
    This class is deprecated. Please use `SentenceWindowRetriever` instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The class `SentenceWindowRetrieval` is deprecated and will be removed in a future release. "
            "Please use `SentenceWindowRetriever` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
