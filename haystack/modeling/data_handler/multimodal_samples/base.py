from typing import Any, Union, Optional, List, Dict

import logging

logger = logging.getLogger(__name__)


class Sample:
    def __init__(self, id: str, data: Dict[str, Any]):
        """
        A single training/test sample. This should contain the input and the label. It should contain also the original
        human readable data source (like the sentence, or the path to the original image).
        Over the course of data preprocessing, this object will be populated with processed versions of the data.

        :param id: The id of the sample
        :param data: A dictionary containing various human readable fields (e.g. text, label).
        """
        self.id = id
        self.data = data


# FIXME Review where this is used and how - raw and samples should get better names
class SampleBasket:
    def __init__(self, id: str, raw: Dict[str, Any], samples: Optional[List[Sample]] = None):
        """
        An object that contains one source text and the one or more samples that will be processed. This
        is needed for tasks like question answering where the source text can generate multiple input - label
        pairs.

        :param id: A unique identifying id.
        :param raw: Contains the various data needed to form a sample. It is ideally in human readable form.
        :param samples: An optional list of Samples used to populate the basket at initialization.
        """
        self.id = id
        self.raw = raw
        self.samples = samples
