# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Union, get_origin, get_args
import logging
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class InputSocket:
    name: str
    type: type
    is_optional: bool = field(init=False)
    sender: Optional[str] = None

    def __post_init__(self):
        self.is_optional = get_origin(self.type) is Union and type(None) in get_args(self.type)


@dataclass
class OutputSocket:
    name: str
    type: type
