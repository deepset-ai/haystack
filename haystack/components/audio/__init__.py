# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {"whisper_local": ["LocalWhisperTranscriber"], "whisper_remote": ["RemoteWhisperTranscriber"]}

if TYPE_CHECKING:
    from .whisper_local import LocalWhisperTranscriber
    from .whisper_remote import RemoteWhisperTranscriber

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
