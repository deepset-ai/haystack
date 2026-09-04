# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from haystack import component


@component
class HelloUsingFutureAnnotations:
    @component.output_types(output=str)
    def run(self, word: str) -> dict[str, str]:
        """Takes a string in input and returns "Hello, <string>!"in output."""
        return {"output": f"Hello, {word}!"}
