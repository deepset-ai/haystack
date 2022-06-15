HF_PARAMETERS_BY_MODEL = {
    "bert": {"prefix": "Bert"},
    "xlm.*roberta": {"prefix": "XLMRoberta"},
    "roberta.*xml": {"prefix": "XLMRoberta"},
    "bigbird": {"prefix": "BigBird"},
    "roberta": {"prefix": "Roberta"},
    "codebert.*mlm": {"prefix": "Roberta"},
    "mlm.*codebert": {"prefix": "Roberta"},
    "camembert": {"prefix": "Camembert"},
    "umberto": {"prefix": "Camembert"},
    "albert": {"prefix": "Albert"},
    "distilbert": {
        "prefix": "DistilBert",
        "sequence_summary_config": {"summary_last_dropout": 0, "summary_type": "first", "summary_activation": "tanh"},
    },
    "xlnet": {"prefix": "XLNet", "sequence_summary_config": {"summary_last_dropout": 0}},
    "electra": {
        "prefix": "Electra",
        "sequence_summary_config": {
            "summary_last_dropout": 0,
            "summary_type": "first",
            "summary_activation": "gelu",
            "summary_use_proj": False,
        },
    },
    "word2vec": {"prefix": "WordEmbedding_LM"},
    "glove": {"prefix": "WordEmbedding_LM"},
    "minilm": {"prefix": "Bert"},
    "deberta-v2": {
        "prefix": "DebertaV2",
        "sequence_summary_config": {
            "summary_last_dropout": 0,
            "summary_type": "first",
            "summary_activati": "tanh",
            "summary_use_proj": False,
        },
    },
    "data2vec-vision": {
        "prefix": "Data2VecVision",
    }
}

HF_MODEL_TYPES = {
    "bert": "Bert",
    "albert": "Albert",
    "roberta": "Roberta",
    "xlm-roberta": "XLMRoberta",
    "distilbert": "DistilBert",
    "xlnet": "XLNet",
    "electra": "Electra",
    "camembert": "Camembert",
    "big_bird": "BigBird",
    "deberta-v2": "DebertaV2",
    "data2vec-vision":  "Data2VecVision",
}

HF_MODEL_STRINGS_HINTS = {
    "xlm.*roberta|roberta.*xlm": "XLMRoberta",
    "bigbird": "BigBird",
    "roberta": "Roberta",
    "codebert": "Roberta",
    "camembert": "Camembert",
    "albert": "Albert",
    "distilbert": "DistilBert",
    "bert": "Bert",
    "xlnet": "XLNet",
    "electra": "Electra",
    "word2vec": "WordEmbedding_LM",
    "glove": "WordEmbedding_LM",
    "minilm": "Bert",
    "dpr-question_encoder": "DPRQuestionEncoder",
    "dpr-ctx_encoder": "DPRContextEncoder",
    "data2vec-vision":  "Data2VecVision",
}

KNOWN_LANGUAGES = ("german", "english", "chinese", "indian", "french", "polish", "spanish", "multilingual")
KNOWN_LANGUAGE_SPECIFIC_MODELS = (("camembert", "french"), ("umberto", "italian"))





TOKENIZERS_PARAMS = {
    "Albert": {"keep_accents": True},
    "XLMRoberta": {},
    "Roberta": {},
    "DistilBert": {},
    "Bert": {},
    "XLNet": {"keep_accents": True},
    "Electra": {},
    "Camembert": {},
    "DPRQuestionEncoder": {},
    "DPRContextEncoder": {},
    "BigBird": {},
    "DebertaV2": {},
}

TOKENIZERS_MAPPING = {
    "albert": "Albert",
    "xlm-roberta": "XLMRoberta",
    "roberta": "Roberta",
    "distilbert": "DistilBert",
    "bert": "Bert",
    "xlnet": "XLNet",
    "electra": "Electra",
    "camembert": "Camembert",
    "big_bird": "BigBird",
    "deberta-v2": "DebertaV2",
}

TOKENIZERS_STRING_HINTS = {
    "albert": "Albert",
    "bigbird": "BigBird",
    "xlm-roberta": "XLMRoberta",
    "roberta": "Roberta",
    "codebert": "Roberta",
    "camembert": "Camembert",
    "umberto": "Camembert",
    "distilbert": "DistilBert",
    "debertav2": "DebertaV2",
    "debertav3": "DebertaV2",
    "bert": "Bert",
    "xlnet": "XLNet",
    "electra": "Electra",
    "minilm": "Bert",
    "dpr-question_encoder": "DPRQuestionEncoder",
    "dpr-ctx_encoder": "DPRContextEncoder",
}