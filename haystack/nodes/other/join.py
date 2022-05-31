from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Union, Any

from haystack import MultiLabel, Document
from haystack.nodes.base import BaseComponent

class JoinNode(BaseComponent):
    def run(self, query: Optional[str] = None, file_paths: Optional[List[str]] = None, labels: Optional[MultiLabel] = None, documents: Optional[List[Document]] = None, meta: Optional[dict] = None, inputs: Optional[List[dict]] = None, **kwargs) -> Tuple[Dict, str]:
        if inputs:
            return self.run_accumulated(inputs, **kwargs)
        return self.run_accumulated(inputs=[{"query": query, "file_paths": file_paths, "labels": labels, "documents": documents, "meta": meta}], **kwargs)
    
    @abstractmethod
    def run_accumulated(self, inputs: List[dict]) -> Tuple[Dict, str]:
        pass

    def run_batch(self, queries: Optional[Union[str, List[str]]] = None, file_paths: Optional[List[str]] = None, labels: Optional[Union[MultiLabel, List[MultiLabel]]] = None, documents: Optional[Union[List[Document], List[List[Document]]]] = None, meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None, params: Optional[dict] = None, debug: Optional[bool] = None, inputs: Optional[List[dict]] = None, **kwargs) -> Tuple[Dict, str]:
        if inputs:
            return self.run_batch_accumulated(inputs=inputs, **kwargs)
        return self.run_batch_accumulated(inputs=[{"queries": queries, "file_paths": file_paths, "labels": labels, "documents": documents, "meta": meta, "params": params, "debug": debug}], **kwargs)
    
    @abstractmethod
    def run_batch_accumulated(self, inputs: List[dict]) -> Tuple[Dict, str]:
        pass