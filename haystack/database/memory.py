from typing import Any, Dict, List, Optional, Union, Tuple
from uuid import UUID

from haystack.database.base import BaseDocumentStore, Document


class InMemoryDocumentStore(BaseDocumentStore):
    """
        In-memory document store
    """

    def __init__(self, embedding_field: Optional[str] = None):
        self.docs = {}  # type: Dict[UUID, Any]
        self.doc_tags = {}  # type: Dict[str, Any]
        self.index = None

    def write_documents(self, documents: Union[List[dict], List[Document]], index:Optional[str]=None):
        """
        Indexes documents for later queries.


        :param documents: List of dictionaries in the format {"text": "<the-actual-text>"}.
                          Optionally, you can also supply "tags": ["one-tag", "another-one"]
                          or additional meta data via "meta": {"name": "<some-document-name>, "author": "someone", "url":"some-url" ...}
                          TODO update

        :return: None
        """

        if index: raise NotImplementedError("Custom index not yet supported for this operation")

        if documents is None:
            return

        for document in documents:
            _doc = document.copy()
            if type(document) == dict:
                _doc = Document.from_dict(_doc)

            self.docs[_doc.id] = _doc

            #TODO fix tags after id refactoring
            tags = document.get("tags", [])
            self._map_tags_to_ids(_doc.id, tags)

    def _map_tags_to_ids(self, hash: str, tags: List[str]):
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, dict):
                    tag_keys = tag.keys()
                    for tag_key in tag_keys:
                        tag_values = tag.get(tag_key, [])
                        if tag_values:
                            for tag_value in tag_values:
                                comp_key = str((tag_key, tag_value))
                                if comp_key in self.doc_tags:
                                    self.doc_tags[comp_key].append(hash)
                                else:
                                    self.doc_tags[comp_key] = [hash]

    def get_document_by_id(self, id: Union[str, UUID], index: Optional[str]=None) -> Document:
        if index: raise NotImplementedError("Custom index not yet supported for this operation")
        return self.docs[id]

    def _convert_memory_hit_to_document(self, hit: Dict[str, Any], doc_id: Optional[str] = None) -> Document:
        document = Document(
            id=doc_id,
            text=hit.get("text", None),
            meta=hit.get("meta", {}),
            query_score=hit.get("query_score", None),
        )
        return document

    def query_by_embedding(self,
                           query_emb: List[float],
                           filters: Optional[dict] = None,
                           top_k: int = 10,
                           index: Optional[str] = None) -> List[Document]:

        from numpy import dot
        from numpy.linalg import norm

        if filters:
            raise NotImplementedError("Setting `filters` is currently not supported in "
                                      "InMemoryDocumentStore.query_by_embedding(). Please remove filters or "
                                      "use a different DocumentStore (e.g. ElasticsearchDocumentStore).")

        if index: raise NotImplementedError("Custom index not yet supported for this operation")

        if query_emb is None:
            return []

        candidate_docs = []
        for idx, doc in self.docs.items():
            doc.query_score = dot(query_emb, doc.embedding) / (
                norm(query_emb) * norm(doc.embedding)
            )
            candidate_docs.append(doc)

        return sorted(candidate_docs, key=lambda x: x.query_score, reverse=True)[0:top_k]

    def update_embeddings(self, retriever):
        """
        Updates the embeddings in the the document store using the encoding model specified in the retriever.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

        :param retriever: Retriever
        :return: None
        """
        #TODO
        raise NotImplementedError("update_embeddings() is not yet implemented for this DocumentStore")

    def get_document_ids_by_tags(self, tags: Union[List[Dict[str, Union[str, List[str]]]], Dict[str, Union[str, List[str]]]]) -> List[str]:
        """
        The format for the dict is {"tag-1": "value-1", "tag-2": "value-2" ...}
        The format for the dict is {"tag-1": ["value-1","value-2"], "tag-2": ["value-3]" ...}
        """
        if not isinstance(tags, list):
            tags = [tags]
        result = self._find_ids_by_tags(tags)
        return result

    def _find_ids_by_tags(self, tags: List[Dict[str, Union[str, List[str]]]]):
        result = []
        for tag in tags:
            tag_keys = tag.keys()
            for tag_key in tag_keys:
                tag_values = tag.get(tag_key, None)
                if tag_values:
                    for tag_value in tag_values:
                        comp_key = str((tag_key, tag_value))
                        doc_ids = self.doc_tags.get(comp_key, [])
                        for doc_id in doc_ids:
                            result.append(self.docs.get(doc_id))
        return result

    def get_document_count(self) -> int:
        return len(self.docs.items())

    def get_all_documents(self) -> List[Document]:
        return list(self.docs.values())
