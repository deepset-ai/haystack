# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests for model device placement on GPU hardware (CUDA or Intel XPU).

Covers device resolution (auto-selection and the ``HAYSTACK_XPU_ENABLED`` gate) and
explicit placement of a SentenceTransformers embedder, asserting against a real
accelerator. The available device is auto-detected, and the tests skip when no GPU is
present, so they are safe to run in CPU-only CI.
"""

import pytest

from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.utils import ComponentDevice

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _available_gpu_devices() -> list[str]:
    """
    Return the torch device strings for the GPUs available on this machine.

    Returns an empty list on CPU-only machines, which causes the GPU tests to be skipped.
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
    reason="No GPU (CUDA or XPU) available. These tests require an accelerator.",
)


DOCUMENTS = [
    Document(content="My name is Jean and I live in Paris."),
    Document(content="My name is Mark and I live in Berlin."),
    Document(content="My name is Mario and I live in the capital of Italy."),
    Document(content="My name is Giorgio and I live in Rome."),
    Document(content="The Eiffel Tower is a wrought-iron lattice tower in Paris, France."),
    Document(content="Rome is the capital city of Italy and was the center of the Roman Empire."),
]


@requires_gpu
def test_auto_select_resolves_to_gpu(del_hf_env_vars, monkeypatch):
    """
    Device resolution against a real accelerator.

    1. With no ``device=`` argument, a component resolves to the GPU via
       ``_get_default_device()`` (priority CUDA > XPU > MPS > CPU).

    2. ``HAYSTACK_XPU_ENABLED`` gates only the XPU branch, so ``"false"`` forces a CPU
       fallback on an XPU-only host while CUDA hosts stay on CUDA.
    """
    import torch

    # Start from the default (unset) env state so auto-selection is unbiased.
    monkeypatch.delenv("HAYSTACK_XPU_ENABLED", raising=False)

    cuda = torch.cuda.is_available()
    expected_type = "cuda" if cuda else "xpu"

    # --- 1. Real model, no device= -> lands on the auto-selected GPU ---
    embedder = SentenceTransformersDocumentEmbedder(model=EMBEDDING_MODEL)
    embedder.warm_up()
    assert embedder.device.to_torch_str().split(":")[0] == expected_type
    assert embedder.embedding_backend.model.device.type == expected_type

    # --- 2. HAYSTACK_XPU_ENABLED gates ONLY the XPU branch ---
    monkeypatch.setenv("HAYSTACK_XPU_ENABLED", "false")
    resolved = ComponentDevice.resolve_device(None).to_torch_str().split(":")[0]
    if cuda:
        # CUDA is not gated by this var — still selected.
        assert resolved == "cuda"
    else:
        # XPU disabled and no CUDA/MPS on an Intel GPU host -> CPU fallback.
        assert resolved == "cpu"


@requires_gpu
@gpu_device
def test_embedder_places_model_on_gpu(device_str, del_hf_env_vars):
    """The SentenceTransformers embedders must load their model onto the requested GPU."""
    device = ComponentDevice.from_str(device_str)

    doc_embedder = SentenceTransformersDocumentEmbedder(model=EMBEDDING_MODEL, device=device)
    doc_embedder.warm_up()

    assert doc_embedder.device.to_torch_str() == device_str
    model_device = doc_embedder.embedding_backend.model.device
    assert model_device.type == device_str.split(":")[0]

    result = doc_embedder.run(documents=DOCUMENTS)
    embedded = result["documents"]
    assert len(embedded) == len(DOCUMENTS)
    assert all(doc.embedding is not None for doc in embedded)
    assert len(embedded[0].embedding) == 384  # all-MiniLM-L6-v2 embedding dim
