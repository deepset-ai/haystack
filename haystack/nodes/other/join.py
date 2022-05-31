from abc import abstractmethod
from typing import Optional, List, Tuple, Dict, Union, Any
import warnings

from haystack import MultiLabel, Document, Answer
from haystack.nodes.base import BaseComponent


class JoinNode(BaseComponent):
    def run(
        self,
        inputs: Optional[List[dict]] = None,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
        answers: Optional[List[Answer]] = None,
        **kwargs
    ) -> Tuple[Dict, str]:  # type: ignore
        if inputs:
            return self.run_accumulated(inputs, **kwargs)
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
            **kwargs
        )

    @abstractmethod
    def run_accumulated(self, inputs: List[dict]) -> Tuple[Dict, str]:
        pass

    def run_batch(
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
        **kwargs
    ) -> Tuple[Dict, str]:  # type: ignore
        if inputs:
            return self.run_batch_accumulated(inputs=inputs, **kwargs)
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
            **kwargs
        )

    @abstractmethod
    def run_batch_accumulated(self, inputs: List[dict]) -> Tuple[Dict, str]:
        pass
