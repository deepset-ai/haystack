# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import time

import pytest

from haystack.tools.tool_cache import InMemoryToolCache, ToolCache, ToolCacheStats, make_cache_key


class TestToolCacheStats:
    def test_default_hits_and_misses_are_zero(self):
        stats = ToolCacheStats()
        assert stats.hits == 0
        assert stats.misses == 0

    def test_calls_saved_equals_hits(self):
        stats = ToolCacheStats(hits=3, misses=2)
        assert stats.calls_saved == 3

    def test_to_dict(self):
        stats = ToolCacheStats(hits=3, misses=2)
        assert stats.to_dict() == {"hits": 3, "misses": 2, "calls_saved": 3}


class TestInMemoryToolCache:
    def test_get_on_missing_key_returns_none(self):
        backend = InMemoryToolCache()
        assert backend.get("missing") is None

    def test_set_then_get_returns_value_and_stored_at(self):
        backend = InMemoryToolCache()
        backend.set("key", "value", 123.0)
        assert backend.get("key") == ("value", 123.0)

    def test_clear_empties_store(self):
        backend = InMemoryToolCache()
        backend.set("key", "value", 123.0)
        backend.clear()
        assert backend.get("key") is None


class TestMakeCacheKey:
    def test_same_args_different_key_order_produce_same_key(self):
        key1 = make_cache_key("weather", {"city": "Berlin", "unit": "C"})
        key2 = make_cache_key("weather", {"unit": "C", "city": "Berlin"})
        assert key1 == key2

    def test_different_args_produce_different_keys(self):
        key1 = make_cache_key("weather", {"city": "Berlin"})
        key2 = make_cache_key("weather", {"city": "Paris"})
        assert key1 != key2

    def test_non_json_serializable_args_fall_back_to_str(self):
        class Unserializable:
            def __str__(self):
                return "unserializable-repr"

        key = make_cache_key("weather", {"obj": Unserializable()})
        assert isinstance(key, str)


class TestToolCache:
    def test_init_invalid_scope_raises(self):
        with pytest.raises(ValueError, match="scope must be one of"):
            ToolCache(scope="invalid_scope")

    def test_init_non_positive_ttl_seconds_raises(self):
        with pytest.raises(ValueError, match="ttl_seconds must be > 0"):
            ToolCache(ttl_seconds=0)

        with pytest.raises(ValueError, match="ttl_seconds must be > 0"):
            ToolCache(ttl_seconds=-1)

    def test_get_on_empty_cache_is_a_miss(self):
        cache = ToolCache()
        hit, value = cache.get("weather", {"city": "Berlin"})
        assert hit is False
        assert value is None
        assert cache.stats.misses == 1
        assert cache.stats.hits == 0

    def test_set_then_get_within_ttl_is_a_hit(self):
        cache = ToolCache(ttl_seconds=300)
        cache.set("weather", {"city": "Berlin"}, "sunny")

        hit, value = cache.get("weather", {"city": "Berlin"})

        assert hit is True
        assert value == "sunny"
        assert cache.stats.hits == 1
        assert cache.stats.misses == 0

    def test_get_after_ttl_expiry_is_a_miss(self, monkeypatch):
        cache = ToolCache(ttl_seconds=10)

        monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
        cache.set("weather", {"city": "Berlin"}, "sunny")

        monkeypatch.setattr(time, "monotonic", lambda: 1011.0)
        hit, value = cache.get("weather", {"city": "Berlin"})

        assert hit is False
        assert value is None
        assert cache.stats.misses == 1
        assert cache.stats.hits == 0

    def test_clear_resets_backend_and_stats(self):
        cache = ToolCache()
        cache.set("weather", {"city": "Berlin"}, "sunny")
        cache.get("weather", {"city": "Berlin"})
        cache.get("weather", {"city": "Paris"})

        cache.clear()

        assert cache.stats.hits == 0
        assert cache.stats.misses == 0
        hit, value = cache.get("weather", {"city": "Berlin"})
        assert hit is False
        assert value is None
