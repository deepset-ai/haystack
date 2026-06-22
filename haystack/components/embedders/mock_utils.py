# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import math
import random
from collections.abc import Callable

# A callable that derives an embedding from the (prepared) text to embed. It receives the text and returns the
# embedding as a list of floats.
EmbeddingFn = Callable[[str], list[float]]


def _l2_normalize(vector: list[float]) -> list[float]:
    """Return the L2-normalized vector, so that mock embeddings behave like real (unit-length) ones."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _deterministic_embedding(text: str, dimension: int) -> list[float]:
    """
    Generate a deterministic, unit-length embedding from the given text.

    The same text always yields the same embedding, and different texts yield different embeddings, which makes mock
    embeddings usable in retrieval pipelines and reproducible across runs and processes. The seed is derived from a
    SHA-256 digest of the text (not the process-salted built-in `hash`) to guarantee cross-process stability.

    :param text: The text to embed.
    :param dimension: The number of dimensions of the resulting embedding.
    :returns: A deterministic, L2-normalized embedding of length `dimension`.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = random.Random(seed)
    vector = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
    return _l2_normalize(vector)


def _coerce_embedding(result: object) -> list[float]:
    """Validate and coerce the output of an `embedding_fn` into a list of floats."""
    if not isinstance(result, (list, tuple)) or not all(isinstance(value, (int, float)) for value in result):
        raise TypeError(f"'embedding_fn' must return a sequence of numbers, got {type(result)}.")
    return [float(value) for value in result]


def _validate_embedding(embedding: list[float]) -> list[float]:
    """Validate that a user-provided fixed embedding is a non-empty sequence of numbers."""
    if not isinstance(embedding, (list, tuple)) or not all(isinstance(value, (int, float)) for value in embedding):
        raise TypeError(f"'embedding' must be a sequence of numbers, got {type(embedding)}.")
    if len(embedding) == 0:
        raise ValueError("'embedding' must not be empty.")
    return [float(value) for value in embedding]


def _estimate_usage(texts: list[str]) -> dict[str, int]:
    """
    Roughly estimate token usage as whitespace-separated word counts.

    This is an approximation (not real tokenization) intended to give downstream code realistic-looking metadata.
    """
    prompt_tokens = sum(len(text.split()) for text in texts)
    return {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens}
