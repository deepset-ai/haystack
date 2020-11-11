from langdetect import detect
import spacy


class QueryExpander:
    def __init__(num_keyword_repeats=3, num_synonyms_per_word=3):
        self.num_keyword_repeats = num_keyword_repeats
        self.num_synonyms_per_word = num_synonyms_per_word
        self.spacy_models = {}
        self.spacy_models["multi"] = spacy.load("xx_ent_wiki_sm")
        self.spacy_models["en"] = spacy.load("en_core_web_sm")

    def remove_stop(query, lang=None):
        if lang == None:
            lang = detect(query)
        if lang in ["en"]:
            keyword_model = self.spacy_models[lang]

        if lang in ["de", "fr", "es", "pt", "it", "nl", "el", "nb", "lt"]:
            try:
                keyword_model = self.spacy_models[lang]
            except:
                try:
                    self.spacy_models[lang] = spacy.load(lang + "_core_news_sm")
                    keyword_model = self.spacy_models[lang]
                except:
                    print('Please install the "' + lang + '_core_news_sm" spacy model')

        if keyword_model == None:
            keyword_model = spacy_models["multi"]

        query = keyword_model(query)
        notStopWords = [
            notStopWords.text for notStopWords in query if not notStopWords.is_stop
        ]
        result = ""
        for xy in range(self.num_keyword_repeats):
            for x in notStopWords:
                result += x + " "

        return result
