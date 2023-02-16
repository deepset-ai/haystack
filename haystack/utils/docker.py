import logging
from typing import List, Union, Optional
from haystack.nodes._json_schema import load_schema


def cache_nltk_model(model: str = "punkt"):
    logging.info("Caching %s model...", model)
    import nltk

    nltk.download(model)


def cache_models(models: Optional[List[str]] = None, use_auth_token: Optional[Union[str, bool]] = None):
    """
    Small function that caches models and other data.
    Used only in the Dockerfile to include these caches in the images.

    :param models: List of Hugging Face model names to cache
    :param use_auth_token: The API token used to download private models from Huggingface.
                           If this parameter is set to `True`, then the token generated when running
                           `transformers-cli login` (stored in ~/.huggingface) will be used.
                           Additional information can be found here
                           https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
    """
    # Backward compat after adding the `model` param
    if models is None:
        models = ["deepset/roberta-base-squad2"]

    # Cache models
    import transformers

    for model_to_cache in models:
        logging.info("Caching %s", model_to_cache)
        transformers.AutoTokenizer.from_pretrained(model_to_cache, use_auth_token=use_auth_token)
        transformers.AutoModel.from_pretrained(model_to_cache, use_auth_token=use_auth_token)


def cache_schema():
    """
    Generate and persist Haystack JSON schema.

    The schema is lazily generated at first usage, but this might not work in Docker containers
    when the user running Haystack doesn't have write permissions on the Python installation. By
    calling this function at Docker image build time, the schema is generated once for all.
    """
    # Calling load_schema() will generate the schema as a side effect
    load_schema()
