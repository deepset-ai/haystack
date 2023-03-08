from typing import Literal, Any, Dict, List, Iterable

import logging


logger = logging.getLogger(__name__)


class StoreError(Exception):
    pass


class DuplicateError(StoreError):
    pass


class MissingPoolError(StoreError):
    pass


class MissingItemError(StoreError):
    pass


class MissingEmbeddingError(StoreError):
    pass


class PoolFullError(StoreError):
    pass


DuplicatePolicy = Literal["skip", "overwrite", "fail"]


class MemoryStore:
    """
    Stores data in-memory, without persisting it anywhere.
    Ephemeral storage that can't (yet) be saved to disk in any way.

    Note: this is a small implementation meant to showcase the contract.
    """

    def __init__(self):
        """
        Initializes the store.
        """
        self.storage = {}

    def has_item(self, id: str) -> bool:
        """
        Checks if this ID exists in the store.

        :param id: the id to find in the store.
        """
        return id in self.storage.keys()

    def get_item(self, id: str) -> Dict[str, Any]:
        """
        Finds a item by ID in the store. Fails if the item is not present.

        :param id: the id of the item to get.
        """
        if not self.has_item(id=id):
            raise MissingItemError(f"ID {id} not found.")
        return self.storage[id]

    def count_items(self, filters: Dict[str, Any]) -> int:
        """
        Returns the number of how many items match the given filters.
        Pass filters={} to count all items in the store.

        :param filters: the filters to apply to the items list.
        """
        # TODO apply filters
        return len(self.storage.keys())

    def get_ids(self, filters: Dict[str, Any]) -> Iterable[str]:
        """
        Returns only the IDs of the items that match the filters provided.

        :param filters: the filters to apply to the item list.
        """
        # TODO apply filters
        for id in self.storage.keys():
            yield id

    def get_items(self, filters: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        """
        Returns the items that match the filters provided.

        :param filters: the filters to apply to the item list.
        """
        # TODO apply filters
        for id in self.get_ids(filters=filters):
            yield self.storage[id]

    def write_items(self, items: Iterable[Dict[str, Any]], duplicates: DuplicatePolicy) -> None:
        """
        Writes items into the store.

        :param items: a list of dictionaries.
        :param duplicates: items with the same ID count as duplicates. When duplicates are met,
            the store can:
             - skip: keep the existing item and ignore the new one.
             - overwrite: remove the old item and write the new one.
             - fail: an error is raised
        :raises DuplicateError: Exception trigger on duplicate item
        :return: None
        """
        for item in items:
            if self.has_item(item["id"]):
                if duplicates == "fail":
                    raise DuplicateError(f"ID {item['id']} already exists.")
                elif duplicates == "skip":
                    logger.warning("ID '%s' already exists", item["id"])
            self.storage[item["id"]] = item

    def delete_items(self, ids: List[str], fail_on_missing_item: bool = False) -> None:
        """
        Deletes all ids from the given pool.

        :param ids: the ids to delete
        :param fail_on_missing_item: fail if the id is not found, log ignore otherwise
        """
        for id in ids:
            if not self.has_item(id=id):
                if fail_on_missing_item:
                    raise MissingItemError(f"ID {id} not found, cannot delete it.")
                logger.info(f"ID {id} not found, cannot delete it.")
            del self.storage[id]
