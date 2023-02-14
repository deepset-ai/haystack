from abc import abstractmethod
from typing import Optional, List, Tuple, Dict, Union, Any
import warnings

from haystack import MultiLabel, Document, Answer
from haystack.nodes.base import BaseComponent


class JoinNode(BaseComponent):
    outgoing_edges: int = 1

    def run(  # type: ignore
        self,
        inputs: Optional[List[dict]] = None,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
        answers: Optional[List[Answer]] = None,
        top_k_join: Optional[int] = None,
    ) -> Tuple[Dict, str]:
        if inputs:
            return self.run_accumulated(inputs, top_k_join=top_k_join)
        warnings.warn("You are using a JoinNode with only one input. This is usually equivalent to a no-op.")
        return self.run_accumulated(
            inputs=[
                {
                    "query": query,
                    "file_paths": file_paths,
                    "labels": labels,
                    "documents": documents,
                    "meta": meta,
                    "answers": answers,
                }
            ],
            top_k_join=top_k_join,
        )

    @abstractmethod
    def run_accumulated(self, inputs: List[dict], top_k_join: Optional[int] = None) -> Tuple[Dict, str]:
        pass

    def run_batch(  # type: ignore
        self,
        inputs: Optional[List[dict]] = None,
        queries: Optional[Union[str, List[str]]] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[Union[MultiLabel, List[MultiLabel]]] = None,
        documents: Optional[Union[List[Document], List[List[Document]]]] = None,
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
        answers: Optional[List[Answer]] = None,
        top_k_join: Optional[int] = None,
    ) -> Tuple[Dict, str]:
        if inputs:
            return self.run_batch_accumulated(inputs=inputs, top_k_join=top_k_join)
        warnings.warn("You are using a JoinNode with only one input. This is usually equivalent to a no-op.")
        return self.run_batch_accumulated(
            inputs=[
                {
                    "queries": queries,
                    "file_paths": file_paths,
                    "labels": labels,
                    "documents": documents,
                    "meta": meta,
                    "params": params,
                    "debug": debug,
                    "answers": answers,
                }
            ],
            top_k_join=top_k_join,
        )

    @abstractmethod
    def run_batch_accumulated(self, inputs: List[dict], top_k_join: Optional[int] = None) -> Tuple[Dict, str]:
        pass
