# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from canals import component


def make_component(input=Any, output=Any):
    @component
    class Component:
        @component.return_types(value=output)
        def run(self, value: input):
            return {"value": value}

    return Component()
