# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from haystack.hooks.tool_result_offloading import FileSystemToolResultStore


class TestFileSystemToolResultStore:
    def test_write_returns_path_and_persists_content(self, tmp_path):
        store = FileSystemToolResultStore(root=tmp_path)
        reference = store.write(key="a.txt", content="hello")
        assert reference == str(tmp_path / "a.txt")
        assert Path(reference).read_text(encoding="utf-8") == "hello"

    def test_write_creates_missing_directories(self, tmp_path):
        store = FileSystemToolResultStore(root=tmp_path / "nested" / "dir")
        reference = store.write(key="a.txt", content="hi")
        assert Path(reference).read_text(encoding="utf-8") == "hi"

    def test_write_allows_nested_keys_within_root(self, tmp_path):
        store = FileSystemToolResultStore(root=tmp_path)
        reference = store.write(key="sub/dir/a.txt", content="ok")
        assert Path(reference).read_text(encoding="utf-8") == "ok"

    def test_write_rejects_parent_traversal_key(self, tmp_path):
        store = FileSystemToolResultStore(root=tmp_path / "root")
        with pytest.raises(ValueError, match="outside the store root"):
            store.write(key="../escape.txt", content="x")
        assert not (tmp_path / "escape.txt").exists()

    def test_write_rejects_absolute_key(self, tmp_path):
        store = FileSystemToolResultStore(root=tmp_path / "root")
        with pytest.raises(ValueError, match="outside the store root"):
            store.write(key=str(tmp_path / "outside.txt"), content="x")

    def test_read_round_trips_written_content(self, tmp_path):
        store = FileSystemToolResultStore(root=tmp_path)
        reference = store.write(key="a.txt", content="round trip")
        assert store.read(reference) == "round trip"

    def test_to_dict_from_dict_roundtrip(self, tmp_path):
        store = FileSystemToolResultStore(root=tmp_path)
        restored = FileSystemToolResultStore.from_dict(store.to_dict())
        assert restored.root == Path(tmp_path)
