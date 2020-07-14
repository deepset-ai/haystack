from typing import Any, Dict, List, Optional, Union, Tuple

from haystack.database.base import BaseDocumentStore, Document


class InMemoryDocumentStore(BaseDocumentStore):
    """
        In-memory document store
    """

    def __init__(self, embedding_field: Optional[str] = None):
        self.docs = {}  # type: Dict[str, Any]
        self.doc_tags = {}  # type: Dict[str, Any]
        self.embedding_field = embedding_field
        self.index = None

    def write_documents(self, documents: List[dict]):
        """
        Indexes documents for later queries.

        :param documents: List of dictionaries in the format {"text": "<the-actual-text>"}.
                          Optionally, you can also supply "tags": ["one-tag", "another-one"]
                          or additional meta data via "meta": {"name": "<some-document-name>, "author": "someone", "url":"some-url" ...}

        :return: None
        """
        import hashlib

        if documents is None:
            return

        for document in documents:
            text = document["text"]
            if "meta" not in document.keys():
                document["meta"] = {}
            for k, v in document.items():  # put additional fields other than text in meta
                if k not in ["text", "meta", "tags"]:
                    document["meta"][k] = v

            if not text:
                raise Exception("A document cannot have empty text field.")

            hash = hashlib.md5(text.encode("utf-8")).hexdigest()

            self.docs[hash] = document

            tags = document.get("tags", [])

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

        if self.embedding_field is None:
            raise Exception(
                "To use query_by_embedding() 'embedding field' must "
                "be specified when initializing the document store."
            )

        if query_emb is None:
            return []

        candidate_docs = []
        for idx, hit in self.docs.items():
            hit["query_score"] = dot(query_emb, hit[self.embedding_field]) / (
                norm(query_emb) * norm(hit[self.embedding_field])
            )
            _doc = self._convert_memory_hit_to_document(hit=hit, doc_id=idx)
            candidate_docs.append(_doc)

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
        return [
            Document(id=item[0], text=item[1]["text"], meta=item[1].get("meta", {}))
            for item in self.docs.items()
        ]
