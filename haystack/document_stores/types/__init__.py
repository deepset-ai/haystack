# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from .policy import DuplicatePolicy
from .protocol import DocumentStore

__all__ = ["DocumentStore", "DuplicatePolicy"]
