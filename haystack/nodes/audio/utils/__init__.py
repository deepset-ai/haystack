from haystack.nodes.audio.utils.text_to_speech import TextToSpeech
from haystack.nodes.audio.utils.speech_transcriber import BaseSpeechTranscriber, Wav2VecTranscriber
from haystack.nodes.audio.utils.forced_aligner import BaseTranscriptAligner, AeneasTranscriptAligner


from haystack.errors import AudioNodeError


TRANSCRIPT_ALIGNERS = {
    class_.__name__.replace("TranscriptAligner", "").lower(): class_ 
    for class_ in BaseTranscriptAligner.__subclasses__()
}

SPEECH_TRANSCRIBERS = {
    class_.__name__.replace("Transcriber", "").lower(): class_ 
    for class_ in BaseSpeechTranscriber.__subclasses__()
}



def get_transcript_aligner(name: str):
    """
    Returns the implementation of the transcript aligner by name.

    For example, 'aeneas' will return the class `AeneasTranscriptAligner`.
    """
    try:
        return TRANSCRIPT_ALIGNERS[name]
    except IndexError as e:
        raise AudioNodeError(
            f"Transcript aligner implementation for '{name}' not found. "
            f"Available transcript aligner implementations: {', '.join(TRANSCRIPT_ALIGNERS.keys())}"
    )


def get_speech_transcriber(name: str):
    """
    Returns the implementation of the transcriber by name.

    For example, 'wav2vec' will return the class `Wav2VecTranscriber`.
    """
    try:
        return SPEECH_TRANSCRIBERS[name]
    except IndexError as e:
        raise AudioNodeError(
            f"Transcriber implementation for '{name}' not found. "
            f"Available transcriber implementations: {', '.join(SPEECH_TRANSCRIBERS.keys())}"
    )