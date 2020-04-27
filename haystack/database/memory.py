from haystack.database.base import BaseDocumentStore, Document


class InMemoryDocumentStore(BaseDocumentStore):
    """
        In-memory document store
    """

    def __init__(self):
        self.docs = {}
        self.doc_tags = {}

    def write_documents(self, documents):
        import hashlib
        for document in documents:
            name = document.get("name", None)
            text = document.get("text", None)

            if name is None or text is None:
                continue

            signature = name + text
            hash = hashlib.md5(signature.encode("utf-8")).hexdigest()

            self.docs[hash] = document

    def get_document_by_id(self, id):
        return self.docs[id]

    def get_document_ids_by_tags(self, tags):
        """
        The format for the dict is {"tag-1": "value-1", "tag-2": "value-2" ...}
        The format for the dict is {"tag-1": ["value-1","value-2"], "tag-2": ["value-3]" ...}
        """
        pass

    def get_document_count(self):
        return len(self.docs.items())

    def get_all_documents(self):
        return [Document(id=item[0], text=item[1]['text'], name=item[1]['name']) for item in self.docs.items()]