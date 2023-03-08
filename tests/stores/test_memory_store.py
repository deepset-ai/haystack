import pytest
from canals.store import MemoryStore, MissingItemError


def test_store():
    store = MemoryStore()

    assert not store.has_item("0")
    with pytest.raises(MissingItemError):
        store.get_item("0")
    store.delete_items(["0"])
    with pytest.raises(MissingItemError):
        store.delete_items(["0"], fail_on_missing_item=True)

    assert store.count_items() == 0
    assert store.get_ids() == []
    assert store.get_items() == []

    with pytest.raises(ValueError, match="must have an 'id' field"):
        store.write_items([{"key": "value"}])

    dictionary = {"id": "0", "key": "value"}
    store.write_items([dictionary])

    assert store.has_item("0")
    assert store.get_item("0") == dictionary
    assert store.count_items() == 1
    assert store.get_ids() == ["0"]
    assert store.get_items() == [dictionary]

    store.delete_items(["0"])
    store.delete_items(["0"])
    with pytest.raises(MissingItemError):
        store.delete_items("0", fail_on_missing_item=True)

    assert not store.has_item("0")
    with pytest.raises(MissingItemError):
        store.get_item("0")
    assert store.count_items() == 0
    assert store.get_ids() == []
    assert store.get_items() == []
