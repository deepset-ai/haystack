from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class Memory(ABC):
    """
    Abstract base class for memory management in an Agent.
    """

    @abstractmethod
    def load(self, keys: Optional[List[str]] = None, **kwargs) -> Any:
        """
        Load the context of this model run from memory.

        :param keys: Optional list of keys to specify the data to load.
        :return: The loaded data.
        """

    @abstractmethod
    def save(self, data: Dict[str, Any]) -> None:
        """
        Save the context of this model run to memory.

        :param data: A dictionary containing the data to save.
        """

    @abstractmethod
    def clear(self) -> None:
        """
        Clear memory contents.
        """
