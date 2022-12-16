import logging
from typing import List, Union, Optional


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

    # download punkt tokenizer
    logging.info("Caching punkt data")
    import nltk

    nltk.download("punkt")

    # Cache models
    import transformers

    for model_to_cache in models:
        logging.info("Caching %s", model_to_cache)
        transformers.AutoTokenizer.from_pretrained(model_to_cache, use_auth_token=use_auth_token)
        transformers.AutoModel.from_pretrained(model_to_cache, use_auth_token=use_auth_token)
