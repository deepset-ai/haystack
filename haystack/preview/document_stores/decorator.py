import logging


logger = logging.getLogger(__name__)


class _Store:
    """
    Marks a class as an Haystack Store.
    All classes decorated with @store will be registered here and can be used in Haystack Pipelines.
    """

    def __init__(self):
        self.registry = {}

    def _decorate(self, cls):
        cls.__haystack_store__ = True

        if cls.__name__ in self.registry:
            logger.error(
                "Store %s is already registered. Previous imported from '%s', new imported from '%s'",
                cls.__name__,
                self.registry[cls.__name__],
                cls,
            )

        self.registry[cls.__name__] = cls
        logger.debug("Registered Store %s", cls)

        return cls

    def __call__(self, cls=None):
        if cls:
            return self._decorate(cls)

        return self._decorate


store = _Store()
