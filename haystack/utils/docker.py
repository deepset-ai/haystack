import logging


def cache_models(models=None):
    """
    Small function that caches models and other data.
    Used only in the Dockerfile to include these caches in the images.
    """
    # Backward compat after adding the `model` param
    if models is None:
        models = ["deepset/roberta-base-squad2"]

    # download punkt tokenizer
    logging.info("Caching punkt data")
    import nltk

    nltk.download("punkt", download_dir="/root/nltk_data")

    # Cache models
    import transformers

    for model_to_cache in models:
        logging.info(f"Caching {model_to_cache}")
        transformers.AutoTokenizer.from_pretrained(model_to_cache)
        transformers.AutoModel.from_pretrained(model_to_cache)
