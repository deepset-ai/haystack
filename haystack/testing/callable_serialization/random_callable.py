# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


def callable_to_deserialize(hello: str) -> str:
    """
    A function to test callable deserialization.
    """
    return f"{hello}, world!"
