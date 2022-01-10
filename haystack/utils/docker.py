
def cache_models():
    """
    Small function that caches models and other data.
    Used only in the Dockerfile to include these caches in the images.
    """
    # download punkt tokenizer
    import nltk
    nltk.download('punkt', download_dir='/usr/nltk_data')
    
    # Cache roberta-base-squad2 model
    import transformers
    model_to_cache='deepset/roberta-base-squad2'
    transformers.AutoTokenizer.from_pretrained(model_to_cache)
    transformers.AutoModel.from_pretrained(model_to_cache)