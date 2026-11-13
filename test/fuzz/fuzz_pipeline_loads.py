# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Fuzz target for ``Pipeline.loads`` — deserializing untrusted serialized pipelines.

Loading a serialized pipeline is an explicit attack surface (see SECURITY.md), so this
target feeds arbitrary fuzzer bytes through the YAML unmarshaller and ``from_dict``.
"""

import sys

import atheris

with atheris.instrument_imports():
    from haystack import Pipeline
    from haystack.core.errors import DeserializationError, PipelineError

# Exceptions that are a normal reaction to malformed input. Anything else — a crash,
# unbounded recursion, a hang, or an unexpected exception type — is a genuine finding.
_EXPECTED = (DeserializationError, PipelineError, ValueError, TypeError, KeyError)


def TestOneInput(data: bytes) -> None:
    """Feed one fuzzer-generated input to ``Pipeline.loads`` as a YAML document."""
    try:
        Pipeline.loads(data)
    except _EXPECTED:
        pass


def main() -> None:
    """Set up and run the Atheris fuzzing loop."""
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
