def test_module_imports():
    from haystack import Finder
    from haystack.document_store.sql import SQLDocumentStore
    from haystack.preprocessor.cleaning import clean_wiki_text
    from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
    from haystack.reader.farm import FARMReader
    from haystack.reader.transformers import TransformersReader
    from haystack.retriever.sparse import TfidfRetriever
    from haystack.utils import print_answers

    assert Finder is not None
    assert SQLDocumentStore is not None
    assert clean_wiki_text is not None
    assert convert_files_to_dicts is not None
    assert fetch_archive_from_http is not None
    assert FARMReader is not None
    assert TransformersReader is not None
    assert TfidfRetriever is not None
    assert print_answers is not None
