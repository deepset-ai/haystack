from haystack.database.base import BaseDocumentStore, Document


class InMemoryDocumentStore(BaseDocumentStore):
    """
        In-memory document store
    """

    def __init__(self):
        self.docs = {}
        self.doc_tags = {}

    def write_documents(self, documents, tags=None):
        import hashlib

        if documents is None:
            return

        if tags is not None and len(tags) == len(documents):
            documents = zip(documents, tags)
        else:
            documents = zip(documents, [None] * len(documents))

        for document, tag in documents:
            name = document.get("name", None)
            text = document.get("text", None)

            if name is None or text is None:
                continue

            signature = name + text

            hash = hashlib.md5(signature.encode("utf-8")).hexdigest()

            self.docs[hash] = document

            if isinstance(tag, dict):
                print(tag)
                tag_key = tag.keys()
                tag_values = tag.values()
                for tag_value in tag_values:
                    self.doc_tags[str((tag_key, tag_value))] = hash

    def get_document_by_id(self, id):
        return self.docs[id]

    def get_document_ids_by_tags(self, tags):
        """
        The format for the dict is {"tag-1": "value-1", "tag-2": "value-2" ...}
        The format for the dict is {"tag-1": ["value-1","value-2"], "tag-2": ["value-3]" ...}
        """
        if not isinstance(tags, list):
            tags = [tags]
        result = []
        for tag in tags:
            tag_key = tag.keys()
            tag_values = tag.values()
            for tag_value in tag_values:
                doc = self.docs.get(self.doc_tags.get(str((tag_key, tag_value)), None), None)
                if doc:
                    result.append(doc)
        return result

    def get_document_count(self):
        return len(self.docs.items())

    def get_all_documents(self):
        return [Document(id=item[0], text=item[1]['text'], name=item[1]['name'], meta=item[1].get('meta', {})) for item in self.docs.items()]
