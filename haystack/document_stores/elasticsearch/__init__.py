try:
    # Use appropriate ElasticsearchDocumentStore depending on ES client version
    from elasticsearch import VERSION

    if VERSION[0] == 8:
        from .es8 import ElasticsearchDocumentStore  # type: ignore
    else:
        from .es7 import ElasticsearchDocumentStore  # type: ignore

except (ModuleNotFoundError, ImportError):
    # Import ES 7 as default if ES is not installed to raise the error message that elasticsearch extra is needed
    from .es7 import ElasticsearchDocumentStore  # type: ignore
