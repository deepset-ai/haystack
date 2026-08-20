# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end test for a fully local RAG pipeline on GPU (CUDA or Intel XPU).

Indexing and querying both run on-device, using ``HuggingFaceLocalChatGenerator`` for
generation, so the test needs no API key and exercises the GPU path from embedding through
generation. The available device is auto-detected, and the test skips when no GPU is
present, so it is safe to run in CPU-only CI.
"""

import pytest

from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import ComponentDevice

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL = "Qwen/Qwen3-0.6B"


def _available_gpu_devices() -> list[str]:
    """
    Return the torch device strings for the GPUs available on this machine.

    Returns an empty list on CPU-only machines, which causes the GPU test to be skipped.
    """
    try:
        import torch
    except ImportError:
        return []

    devices = []
    if torch.cuda.is_available():
        devices.append("cuda:0")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        devices.append("xpu:0")
    return devices


AVAILABLE_GPU_DEVICES = _available_gpu_devices()

# Parametrize over every GPU backend present so the same assertions run on CUDA and XPU.
gpu_device = pytest.mark.parametrize("device_str", AVAILABLE_GPU_DEVICES)
requires_gpu = pytest.mark.skipif(
    not AVAILABLE_GPU_DEVICES,
    reason="No GPU (CUDA or XPU) available. This test requires an accelerator.",
)


@requires_gpu
@gpu_device
def test_fully_local_rag_pipeline(device_str, tmp_path, del_hf_env_vars):
    """
    Fully local RAG pipeline on GPU — no external API key required.

    indexing:  embed (GPU) → write
    querying:  embed (GPU) → retrieve → prompt → local LLM (GPU) → answer
    """
    device = ComponentDevice.from_str(device_str)
    expected_device_type = device_str.split(":")[0]

    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
        Document(content="My name is Mario and I live in the capital of Italy."),
    ]

    # --- Indexing pipeline ---
    document_store = InMemoryDocumentStore()
    indexing = Pipeline()
    indexing.add_component(
        instance=SentenceTransformersDocumentEmbedder(model=EMBEDDING_MODEL, device=device),
        name="doc_embedder",
    )
    indexing.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
    indexing.connect("doc_embedder", "writer")
    indexing.run({"doc_embedder": {"documents": documents}})

    # Confirm embedder ran on GPU
    assert indexing.get_component("doc_embedder").embedding_backend.model.device.type == expected_device_type

    # --- RAG query pipeline ---
    prompt_template = [
        ChatMessage.from_system("You are a helpful assistant. Answer using only the documents provided."),
        ChatMessage.from_user(
            "Documents:\n{% for doc in documents %}{{ doc.content }}\n{% endfor %}\nQuestion: {{ question }}"
        ),
    ]

    rag = Pipeline()
    rag.add_component(
        instance=SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL, device=device),
        name="text_embedder",
    )
    rag.add_component(
        instance=InMemoryEmbeddingRetriever(document_store=document_store, top_k=2),
        name="retriever",
    )
    rag.add_component(instance=ChatPromptBuilder(template=prompt_template), name="prompt_builder")
    rag.add_component(
        instance=HuggingFaceLocalChatGenerator(
            model=GENERATOR_MODEL,
            device=device,
            generation_kwargs={"max_new_tokens": 50},
        ),
        name="llm",
    )
    rag.connect("text_embedder.embedding", "retriever.query_embedding")
    rag.connect("retriever.documents", "prompt_builder.documents")
    rag.connect("prompt_builder.prompt", "llm.messages")

    # Serialize/deserialize — GPU device must survive a YAML round-trip
    with open(tmp_path / "local_rag_gpu.yaml", "w") as f:
        rag.dump(f)
    with open(tmp_path / "local_rag_gpu.yaml", "r") as f:
        rag = Pipeline.load(f)

    result = rag.run(
        {
            "text_embedder": {"text": "Who lives in Rome?"},
            "prompt_builder": {"question": "Who lives in Rome?"},
        }
    )

    # --- Correctness ---
    replies = result["llm"]["replies"]
    assert len(replies) > 0
    answer_text = replies[0].text
    assert isinstance(answer_text, str) and len(answer_text) > 0
    # The answer should mention Giorgio (the person living in Rome)
    assert "Giorgio" in answer_text or "Rome" in answer_text

    # --- GPU placement: text embedder and LLM both on the requested device ---
    assert rag.get_component("text_embedder").embedding_backend.model.device.type == expected_device_type
    assert rag.get_component("llm").pipeline.device.type == expected_device_type
