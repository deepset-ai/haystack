from typing import Optional, List, Any, Dict

from haystack.agents.memory import Memory


class NoMemory(Memory):
    """
    A memory class that doesn't store any data.
    """

    def load(self, keys: Optional[List[str]] = None, **kwargs) -> str:
        """
        Load an empty dictionary.

        :param keys: Optional list of keys (ignored in this implementation).
        :return: An empty str.
        """
        return ""

    def save(self, data: Dict[str, Any]) -> None:
        """
        Save method that does nothing.

        :param data: A dictionary containing the data to save (ignored in this implementation).
        """
        pass

    def clear(self) -> None:
        """
        Clear method that does nothing.
        """
        pass
