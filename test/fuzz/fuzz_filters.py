# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Fuzz target for ``document_matches_filter`` — evaluating untrusted filter expressions."""

import json
import sys

import atheris

with atheris.instrument_imports():
    from haystack.dataclasses import Document
    from haystack.errors import FilterError
    from haystack.utils.filters import document_matches_filter

# A fixed document is enough; we are fuzzing the filter expression, not the document.
_DOCUMENT = Document(content="the quick brown fox", meta={"page": 1, "name": "fuzz"})

# Normal reactions to malformed filters; anything else is a genuine finding.
_EXPECTED = (FilterError, ValueError, TypeError, KeyError)


def TestOneInput(data: bytes) -> None:
    """Decode fuzzer bytes into a JSON filter dict and evaluate it against a document."""
    fdp = atheris.FuzzedDataProvider(data)
    raw = fdp.ConsumeUnicodeNoSurrogates(fdp.remaining_bytes())
    try:
        filters = json.loads(raw)
    except (ValueError, RecursionError):
        return
    if not isinstance(filters, dict):
        return
    try:
        document_matches_filter(filters, _DOCUMENT)
    except _EXPECTED:
        pass


def main() -> None:
    """Set up and run the Atheris fuzzing loop."""
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
