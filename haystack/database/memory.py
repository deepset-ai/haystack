from typing import Any, Dict, List, Optional, Union, Tuple

from haystack.database.base import BaseDocumentStore, Document


class InMemoryDocumentStore(BaseDocumentStore):
    """
        In-memory document store
    """

    def __init__(self, embedding_field: Optional[str] = None):
        self.docs = {}
        self.doc_tags = {}
        self.embedding_field = embedding_field

    def write_documents(self, documents: List[dict]):
        import hashlib

        if documents is None:
            return

        for document in documents:
            name = document.get("name", None)
            text = document.get("text", None)

            if name is None or text is None:
                continue

            signature = name + text

            hash = hashlib.md5(signature.encode("utf-8")).hexdigest()

            self.docs[hash] = document

            tags = document.get('tags', [])

            self._map_tags_to_ids(hash, tags)

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

    def get_document_by_id(self, id: str) -> Document:
        document = self._convert_memory_hit_to_document(self.docs[id], doc_id=id)
        return document

    def _convert_memory_hit_to_document(self, hit: Tuple[Any, Any], doc_id: Optional[str] = None) -> Document:
        document = Document(
            id=doc_id,
            text=hit[0].get('text', None),
            meta=hit[0].get('meta', {}),
            query_score=hit[1],
        )
        return document

    def query_by_embedding(self, query_emb: List[float], top_k: int = 10, candidate_doc_ids: Optional[List[str]] = None) -> List[Document]:
        from numpy import dot
        from numpy.linalg import norm

        if self.embedding_field is None:
            return []

        if query_emb is None:
            return []

        candidate_docs = [self._convert_memory_hit_to_document(
            (doc, dot(query_emb, doc[self.embedding_field]) / (norm(query_emb) * norm(doc[self.embedding_field]))), doc_id=idx) for idx, doc in self.docs.items()
        ]

        return sorted(candidate_docs, key=lambda x: x.query_score, reverse=True)[0:top_k]

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
        return [Document(id=item[0], text=item[1]['text'], name=item[1]['name'], meta=item[1].get('meta', {})) for item in self.docs.items()]
