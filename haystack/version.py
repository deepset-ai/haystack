# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from importlib import metadata

try:
    __version__ = str(metadata.version("haystack-ai"))
except metadata.PackageNotFoundError:
    __version__ = "main"
