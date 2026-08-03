# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# The functions below must live in a module of their own so that `from __future__ import annotations` applies to them
# and their annotations are stored as strings (postponed evaluation).
from __future__ import annotations

from haystack.dataclasses import Document


def retrieve(query: str, documents: list[Document], top_k: int | None = None) -> list[Document]:
    return documents[:top_k]
