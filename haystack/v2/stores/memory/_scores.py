from typing import Iterable

import logging

import numpy as np


logger = logging.getLogger(__name__)

try:
    import torch
except ImportError as e:
    logger.debug("torch is not installed: local embedding retrieval won't work.")


try:
    from numba import njit
except (ImportError, ModuleNotFoundError) as e:
    logger.debug("Numba not found, replacing njit() with no-op implementation. Enable it with 'pip install numba'.")

    def njit(f):
        return f


# TODO can we process several queries at once? We should be able to!
def get_scores_torch(
    query: np.ndarray, documents: Iterable[np.ndarray], similarity: str, batch_size: int, device: "torch.device"
) -> Iterable[float]:
    """
    Calculate similarity scores between query embedding and a list of documents using torch.

    :param query: Embedding of the query
    :param documents: Embeddings of the documents to compare `query` against.
    :param similarity: the similarity metric to use
    :param batch_size: how many documents to process at once
    :param device: the CUDA device to use
    :returns: list of scores in the same order as the documents
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "torch is not installed: local embedding retrieval can't work. "
            "Install torch with `pip install torch` to fix this error."
        ) from e

    query = torch.tensor(query, dtype=torch.float).to(device)
    if len(query.shape) == 1:
        query = query.unsqueeze(dim=0)

    documents = torch.as_tensor(documents, dtype=torch.float)
    if len(documents.shape) == 1 and documents.shape[0] == 1:
        documents = documents.unsqueeze(dim=0)
    elif len(documents.shape) == 1 and documents.shape[0] == 0:
        return []

    if similarity == "cosine":
        # cosine similarity is just a normed dot product
        query_norm = torch.norm(query, dim=1)
        query = torch.div(query, query_norm)
        documents_norms = torch.norm(documents, dim=1)
        documents = torch.div(documents.T, documents_norms).T

    curr_pos = 0
    scores = []
    while curr_pos < len(documents):
        documents_slice = documents[curr_pos : curr_pos + batch_size]
        documents_slice = documents_slice.to(device)
        with torch.inference_mode():
            slice_scores = torch.matmul(documents_slice, query.T).cpu()
            slice_scores = slice_scores.squeeze(dim=1)
            slice_scores = slice_scores.numpy().tolist()
        scores.extend(slice_scores)
        curr_pos += batch_size

    return scores


# TODO can we process several queries at once? We should be able to!
def get_scores_numpy(query: np.ndarray, documents: np.ndarray, similarity: str) -> Iterable[float]:
    """
    Calculate similarity scores between query embedding and a list of documents using numpy.

    :param query: Embedding of the query
    :param documents: Embeddings of the documents to compare `query` against.
    :param similarity: the similarity metric to use
    :param batch_size: how many documents to process at once
    :param device: the CUDA device to use
    :returns: list of scores in the same order as the documents
    """
    if len(query.shape) == 1:
        query = np.expand_dims(query, 0)

    if len(documents.shape) == 1 and documents.shape[0] == 1:
        documents = documents.unsqueeze(dim=0)  # type: ignore [attr-defined]
    elif len(documents.shape) == 1 and documents.shape[0] == 0:
        return []

    if similarity == "cosine":
        # cosine similarity is just a normed dot product
        query_norm = np.apply_along_axis(np.linalg.norm, 1, query)
        query_norm = np.expand_dims(query_norm, 1)
        query = np.divide(query, query_norm)

        documents_norms = np.apply_along_axis(np.linalg.norm, 1, documents)
        documents_norms = np.expand_dims(documents_norms, 1)
        documents = np.divide(documents, documents_norms)

    scores = np.dot(query, documents.T)[0].tolist()
    return scores


def scale_to_unit_interval(score: float, similarity: str) -> float:
    if similarity == "cosine":
        return (score + 1) / 2
    else:
        return float(expit(score / 100))


@njit
def expit(x: float) -> float:
    return 1 / (1 + np.exp(-x))
