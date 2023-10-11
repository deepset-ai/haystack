# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional
import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class InputSocket:
    name: str
    type: type
    is_optional: bool
    sender: Optional[str] = None


@dataclass
class OutputSocket:
    name: str
    type: type
