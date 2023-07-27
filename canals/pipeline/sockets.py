# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional
import logging
from dataclasses import dataclass

from canals.utils import _type_name


logger = logging.getLogger(__name__)


@dataclass
class InputSocket:
    name: str
    type: type
    is_optional: bool
    sender: Optional[str] = None

    def __str__(self):
        return _type_name(self.type)


@dataclass
class OutputSocket:
    name: str
    type: type

    def __str__(self):
        return _type_name(self.type)
