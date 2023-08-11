from typing import Optional
from dataclasses import dataclass


@dataclass
class Answer:
    answer: Optional[str]
    probability: float
