# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


from haystack.core.component import component


@component
class TextSplitter:
    @component.output_types(output=list[str])
    def run(self, sentence: str):
        """Takes a sentence in input and returns its words in output."""
        return {"output": sentence.split()}
