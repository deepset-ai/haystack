import logging
from typing import List, Optional


from haystack.nodes.base import Document
from haystack.nodes.doc_language_classifier.base import BaseDocumentLanguageClassifier

logger = logging.getLogger(__name__)


try:
    import langdetect
except (ImportError, ModuleNotFoundError) as exc:
    logger.debug(
        "langdetect could not be imported. "
        "Run 'pip install farm-haystack[preprocessing]' or 'pip install langdetect' to fix this issue."
    )
    langdetect = None


class LangdetectDocumentLanguageClassifier(BaseDocumentLanguageClassifier):
    """
    A node based on the lightweight and fast [langdetect library](https://github.com/Mimino666/langdetect) for classifying the language of documents.
    This node detects the language of Documents and adds the output to the Documents metadata.
    The meta field of the Document is a dictionary with the following format:
    ``'meta': {'name': '450_Baelor.txt', 'language': 'en'}``
    - Using the document language classifier, you can directly get predictions with `predict()`.
    - You can route the Documents to different branches depending on their language
      by setting the `route_by_language` parameter to `True` and specifying the `languages_to_route` parameter.
    **Usage example**
    ```python
    ...
    docs = [Document(content="The black dog runs across the meadow")]

    doclangclassifier = LangdetectDocumentLanguageClassifier()
    results = doclangclassifier.predict(documents=docs)

    # print the predicted language
    print(results[0].to_dict()["meta"]["language"]

    **Usage example for routing**
    ```python
    ...
    docs = [Document(content="My name is Ryan and I live in London"),
            Document(content="Mi chiamo Matteo e vivo a Roma")]

    doclangclassifier = LangdetectDocumentLanguageClassifier(
        route_by_language = True,
        languages_to_route = ['en','it','es']
        )
    for doc in docs:
        doclangclassifier.run(doc)
    ```
    """

    def __init__(self, route_by_language: bool = True, languages_to_route: Optional[List[str]] = None):
        """
        :param route_by_language: Sends Documents to a different output edge depending on their language.
        :param languages_to_route: A list of languages in ISO code, each corresponding to a different output edge (see
        [langdetect` documentation](https://github.com/Mimino666/langdetect#languages)).
        """
        if not langdetect:
            raise ImportError(
                "langdetect could not be imported. "
                "Run 'pip install farm-haystack[file-conversion]' or 'pip install langdetect' to fix this issue."
            )
        super().__init__(route_by_language=route_by_language, languages_to_route=languages_to_route)

    def predict(self, documents: List[Document], batch_size: Optional[int] = None) -> List[Document]:
        """
        Detect the language of Documents and add the output to the Documents metadata.
        :param documents: A list of Documents whose language you want to detect.
        :return: List of Documents, where Document.meta["language"] contains the predicted language.
        """
        if len(documents) == 0:
            raise ValueError(
                "LangdetectDocumentLanguageClassifier needs at least one document to predict the language."
            )
        if batch_size is not None:
            logger.warning(
                "LangdetectDocumentLanguageClassifier does not support batch_size. This parameter is ignored."
            )

        documents_with_language = []
        for document in documents:
            try:
                language = langdetect.detect(document.content)
            except langdetect.LangDetectException:
                logger.warning("Langdetect cannot detect the language of document: %s", document)
                language = None
            document.meta["language"] = language
            documents_with_language.append(document)
        return documents_with_language

    def predict_batch(self, documents: List[List[Document]], batch_size: Optional[int] = None) -> List[List[Document]]:
        """
        Detect the Document's language and add the output to the Document's meta data.
        :param documents: A list of lists of Documents to detect language.
        :return: List of lists of Documents, where Document.meta["language"] contains the predicted language
        """
        if len(documents) == 0 or all(len(docs_list) == 0 for docs_list in documents):
            raise ValueError(
                "LangdetectDocumentLanguageClassifier needs at least one document to predict the language."
            )
        if batch_size is not None:
            logger.warning(
                "LangdetectDocumentLanguageClassifier does not support batch_size. This parameter is ignored."
            )
        return [self.predict(documents=docs_list) for docs_list in documents]
