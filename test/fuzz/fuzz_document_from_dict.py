# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Fuzz target for ``Document.from_dict`` — deserializing untrusted document dicts."""

import json
import sys

import atheris

with atheris.instrument_imports():
    from haystack.dataclasses import Document

# Normal reactions to malformed input; anything else is a genuine finding.
_EXPECTED = (ValueError, TypeError, KeyError)


def TestOneInput(data: bytes) -> None:
    """Parse fuzzer bytes as a JSON object and feed it to ``Document.from_dict``."""
    # ``json.loads`` accepts ``bytes`` directly, so the raw fuzzer input *is* the JSON
    # text. This keeps the input domain identical to the seed corpus (see corpus/) and
    # lets libFuzzer mutate JSON directly instead of through a FuzzedDataProvider.
    try:
        obj = json.loads(data)
    except (ValueError, RecursionError, UnicodeDecodeError):
        return
    if not isinstance(obj, dict):
        return
    try:
        Document.from_dict(obj)
    except _EXPECTED:
        pass


def main() -> None:
    """Set up and run the Atheris fuzzing loop."""
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
