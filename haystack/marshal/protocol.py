# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Protocol, Union

# Ellipsis are needed for the type checker, it's safe to disable module-wide
# pylint: disable=unnecessary-ellipsis


class Marshaller(Protocol):
    def marshal(self, dict_: Dict[str, Any]) -> str:
        "Convert a dictionary to its string representation"
        ...

    def unmarshal(self, data_: Union[str, bytes, bytearray]) -> Dict[str, Any]:
        """Convert a marshalled object to its dictionary representation"""
        ...
