from typing import Literal, Any, Dict, List, Optional

import logging


logger = logging.getLogger(__name__)


class StoreError(Exception):
    pass


class DuplicateError(StoreError):
    pass


class MissingItemError(StoreError):
    pass


DuplicatePolicy = Literal["skip", "overwrite", "fail"]


class MemoryStore:
    """
    Stores data in-memory. It's ephemeral and cannot be saved to disk.

    Note: this is a toy implementation meant to showcase the contract.
    Can still be used for small applications and demos.
    """

    def __init__(self):
        """
        Initializes the store.
        """
        self.storage = {}

    def has_item(self, obj_id: str) -> bool:
        """
        Checks if this ID exists in the store.

        :param obj_id: the obj_id to find in the store.
        """
        return obj_id in self.storage.keys()

    def get_item(self, obj_id: str) -> Dict[str, Any]:
        """
        Finds a item by ID in the store. Fails if the item is not present.

        :param obj_id: the obj_id of the item to get.
        """
        if not self.has_item(obj_id=obj_id):
            raise MissingItemError(f"ID {obj_id} not found.")
        return self.storage[obj_id]

    def count_items(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Returns the number of how many items match the given filters.
        Pass filters={} to count all items in the store.

        :param filters: the filters to apply to the items list.
        """
        # TODO apply filters
        return len(self.storage.keys())

    def get_obj_ids(self, filters: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Returns only the IDs of the items that match the filters provobj_ided.

        :param filters: the filters to apply to the item list.
        """
        # TODO apply filters
        return list(self.storage.keys())

    def get_items(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Returns the items that match the filters provobj_ided.

        :param filters: the filters to apply to the item list.
        """
        # TODO apply filters
        return [self.storage[obj_id] for obj_id in self.get_obj_ids(filters=filters)]

    def write_items(self, items: List[Dict[str, Any]], duplicates: DuplicatePolicy = "fail") -> None:
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
        if not isinstance(items, list):
            raise ValueError("Please provobj_ide a list of dictionaries.")
        for item in items:
            if not "id" in item.keys():
                raise ValueError("Objects must have an 'id' field.")

            if self.has_item(item["id"]):
                if duplicates == "fail":
                    raise DuplicateError(f"ID {item['id']} already exists.")
                if duplicates == "skip":
                    logger.warning("ID '%s' already exists", item["id"])
            self.storage[item["id"]] = item

    def delete_items(self, obj_ids: List[str], fail_on_missing_item: bool = False) -> None:
        """
        Deletes all obj_ids from the given pool.

        :param obj_ids: the obj_ids to delete
        :param fail_on_missing_item: fail if the obj_id is not found, log ignore otherwise
        """
        for obj_id in obj_ids:
            if not self.has_item(obj_id=obj_id):
                if fail_on_missing_item:
                    raise MissingItemError(f"ID {obj_id} not found, cannot delete it.")
                logger.info("ID %s not found, cannot delete it.", obj_id)
                return
            del self.storage[obj_id]
