# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from canals import component


@component
class AddFixedValue:
    def __init__(self, add: int = 1):
        self.add = add

    @component.return_types(result=int)
    def run(self, value: int, add: Optional[int] = None):
        if add is None:
            add = self.add
        return {"result": value + add}
