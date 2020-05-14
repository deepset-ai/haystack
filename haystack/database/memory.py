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

    def _map_tags_to_ids(self, hash, tags):
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

    def get_document_by_id(self, id):
        return self.docs[id]

    def get_document_ids_by_tags(self, tags):
        """
        The format for the dict is {"tag-1": "value-1", "tag-2": "value-2" ...}
        The format for the dict is {"tag-1": ["value-1","value-2"], "tag-2": ["value-3]" ...}
        """
        if not isinstance(tags, list):
            tags = [tags]
        result = self._find_ids_by_tags(tags)
        return result

    def _find_ids_by_tags(self, tags):
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

    def get_document_count(self):
        return len(self.docs.items())

    def get_all_documents(self):
        return [Document(id=item[0], text=item[1]['text'], name=item[1]['name'], meta=item[1].get('meta', {})) for item in self.docs.items()]
