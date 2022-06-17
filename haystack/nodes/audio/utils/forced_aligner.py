from abc import abstractmethod, ABC

import json
import logging
import datetime
from pathlib import Path

from aeneas.executetask import ExecuteTask
from aeneas.task import Task


class BaseTranscriptAligner(ABC):
    """
    Aligns an audio file containing speech with its transcription.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def align(self, audio_file: Path, transcript_file: Path):
        pass


class AeneasTranscriptAligner(BaseTranscriptAligner):
    """
    Aligns an audio file with its transcription using traditional forced aligmnent methods.
    Based on AENEAS.
    """

    def __init__(self):
        super().__init__()
        self.task = Task(config_string=u"task_language=eng|is_text_type=plain|os_task_file_format=json")

    def align(self, audio_file: Path, transcript_file: Path):
        logging.debug(f"Aligning {audio_file} with {transcript_file}...")
        self.task.audio_file_path_absolute = str(audio_file.absolute())
        self.task.text_file_path_absolute = str(transcript_file.absolute())

        start = datetime.datetime.now()
        ExecuteTask(self.task).execute()

        logging.debug(f"Alignment complete. It required {datetime.datetime.now() - start} sec.")
        return json.loads(self.task.sync_map.json_string).get("fragments", [])
