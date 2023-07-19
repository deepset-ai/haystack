from typing import List

from haystack.preview import component, Document
from haystack.preview.document_stores import StoreAwareMixin


@component
class WriteToStore(StoreAwareMixin):
    @component.input
    def input(self):
        class Input:
            documents: List[Document]

        return Input

    @component.output
    def output(self):
        class Output:
            ...

        return Output

    def run(self, data):
        self.store.write_documents(data.documents)
        return self.output()
