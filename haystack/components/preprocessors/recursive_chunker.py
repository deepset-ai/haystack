from haystack import component


@component
class RecursiveChunker:
    def __init__(self):
        pass

    def _chunk_text(self, text):
        # some logic to split text into smaller chunks
        return text

    def run(self, documents):
        """
        Split text of documents into smaller chunks recursively.

        :param documents:
        :returns:
            Documents with text split into smaller chunks
        """
        for doc in documents:
            doc.text = self._chunk_text(doc.text)
        return documents
