# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.components.audio.whisper_local import LocalWhisperTranscriber
from haystack.components.audio.whisper_remote import RemoteWhisperTranscriber

__all__ = ["LocalWhisperTranscriber", "RemoteWhisperTranscriber"]
