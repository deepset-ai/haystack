import logging

logger = logging.getLogger(__name__)


class _DocumentStore:
    """
    Marks a class as an Haystack _DocumentStore.
    All classes decorated with @document_store will be registered here and can be used in Haystack Pipelines.
    """

    def __init__(self):
        self.registry = {}

    def _decorate(self, cls):
        cls.__haystack_document_store__ = True

        classname = f"{cls.__module__}.{cls.__name__}"
        if classname in self.registry:
            logger.error(
                "DocumentStore %s is already registered. Previous imported from '%s', new imported from '%s'",
                classname,
                self.registry[classname],
                cls,
            )

        self.registry[classname] = cls
        logger.debug("Registered DocumentStore %s", cls)

        return cls

    def __call__(self, cls=None):
        if cls:
            return self._decorate(cls)

        return self._decorate


document_store = _DocumentStore()
