from haystack.utils.import_utils import safe_import
from haystack.nodes.audio.whisper_transcriber import WhisperTranscriber, WhisperModel

AnswerToSpeech = safe_import(
    "haystack.nodes.audio.answer_to_speech", "AnswerToSpeech", "audio"
)  # Has optional dependencies
DocumentToSpeech = safe_import(
    "haystack.nodes.audio.document_to_speech", "DocumentToSpeech", "audio"
)  # Has optional dependencies
