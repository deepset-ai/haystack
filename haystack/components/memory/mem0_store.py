# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from haystack import default_from_dict, default_to_dict
from haystack.dataclasses.chat_message import ChatMessage
from haystack.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install mem0ai'") as mem0_import:
    from mem0 import MemoryClient


@dataclass
class MemoryConfig:
    """
    Search criteria for memory retrieval operations.

    :param user_id: User identifier for scoping the search
    :param query: Text query to search for
    :param backend_config: Configuration dictionary for Mem0 client
    :param filters: Additional filters to apply on search
    :param top_k: Maximum number of results to return
    """

    user_id: str
    query: Optional[str] = None
    backend_config: Optional[dict[Any, Any]] = None
    filters: Optional[dict[str, Any]] = None
    top_k: Optional[int] = None


class Mem0MemoryStore:
    """
    A memory store implementation using Mem0 as the backend.

    :param api_key: Mem0 API key (if not provided, uses MEM0_API_KEY environment variable)
    :param config: Configuration dictionary for Mem0 client
    :param kwargs: Additional configuration parameters for Mem0 client
    """

    def __init__(self, api_key: Optional[str] = None, memory_config: Optional[MemoryConfig] = None):
        mem0_import.check()
        self.api_key = api_key or os.getenv("MEM0_API_KEY")
        if not self.api_key:
            raise ValueError("Mem0 API key must be provided either as parameter or MEM0_API_KEY environment variable")

        self.memory_config = memory_config
        self.client = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the store configuration to a dictionary."""
        return default_to_dict(self, api_key=self.api_key, config=self.config)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mem0MemoryStore":
        """Deserialize the store from a dictionary."""
        return default_from_dict(cls, data)

    def set_memory_config(
        self,
        user_id: str,
        query: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ):
        """
        Set the memory configuration for the memory store.
        """
        self.memory_config = MemoryConfig(user_id=user_id, query=query, filters=filters, top_k=top_k)
        self.client = (
            MemoryClient.from_config(self.memory_config.backend_config)
            if self.memory_config.backend_config
            else MemoryClient(api_key=self.api_key)
        )

    def add_memories(self, user_id: str, messages: list[ChatMessage]) -> list[str]:
        """
        Add ChatMessage memories to Mem0.

        :param messages: List of ChatMessage objects with memory metadata
        :param user_id: User identifier associated with the memories
        :returns: List of memory IDs for the added messages
        """
        added_ids = []

        for message in messages:
            if not message.text:
                continue
            mem0_message = [{"role": message.role, "content": message.text}]
            print(mem0_message)
            print("META: ", message.meta)

            try:
                # Mem0 primarily uses user_id as the main identifier
                # org_id and session_id are stored in metadata for filtering
                result = self.client.add(messages=mem0_message, user_id=user_id, metadata=message.meta, infer=False)
                # Mem0 returns different response formats, handle both
                memory_id = result.get("id") or result.get("memory_id") or str(result)
                added_ids.append(memory_id)
            except Exception as e:
                raise RuntimeError(f"Failed to add memory message: {e}") from e

        return added_ids

    def search_memories(
        self, query: str, user_id: str, filters: Optional[dict[str, Any]] = None, top_k: int = 10
    ) -> list[ChatMessage]:
        """
        Search for memories in Mem0.

        :param query: Text query to search for
        :param user_id: User identifier for scoping the search
        :param filters: Additional filters to apply on search. For more details on mem0 filters, see https://mem0.ai/docs/search/
        :param top_k: Maximum number of results to return
        :returns: List of ChatMessage memories matching the criteria
        """
        # Prepare filters for Mem0
        search_filters = filters or {}

        mem0_filters = {"AND": [{"user_id": user_id}, search_filters]}

        try:
            if not query:
                results = self.client.get_all(filters=mem0_filters, top_k=top_k)
            else:
                results = self.client.search(query=query, limit=top_k, filters=mem0_filters, user_id=user_id)
            print("RESULTS: ", results)
            memories = [
                ChatMessage.from_assistant(text=result["memory"], meta=result["metadata"]) for result in results
            ]

            return memories

        except Exception as e:
            raise RuntimeError(f"Failed to search memories: {e}") from e

    def update_memories(self, messages: list[ChatMessage]) -> int:
        """
        Update ChatMessage memories in Mem0.

        :param messages: List of ChatMessage memories to update (must have memory_id in meta)
        :returns: Number of records actually updated
        """

        for message in messages:
            memory_id = message.meta.get("memory_id")
            if not memory_id:
                raise ValueError("ChatMessage must have memory_id in meta to be updated")

            metadata = {
                "role": message.role.value,
                "updated_at": datetime.now().isoformat(),
                **{k: v for k, v in message.meta.items() if k not in ["memory_id", "user_id"]},
            }

            try:
                self.client.update(memory_id=memory_id, data=message.text or str(message), metadata=metadata)
            except Exception as e:
                raise RuntimeError(f"Failed to update memory {memory_id}: {e}") from e

    # mem0 doesn't allow passing filter to delete endpoint,
    # we can delete all memories for a user by passing the user_id
    def delete_all_memories(self, user_id: str):
        """
        Delete memory records from Mem0.

        :param user_id: User identifier for scoping the deletion
        """

        try:
            self.client.delete_all(user_id=user_id)
        except Exception as e:
            raise RuntimeError(f"Failed to delete memories for user {user_id}: {e}") from e
