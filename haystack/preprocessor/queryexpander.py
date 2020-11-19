from langdetect import detect
import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator 



class QueryExpander:
    def __init__(num_keyword_repeats=3, num_synonyms_per_word=3):
        self.num_keyword_repeats = num_keyword_repeats
        self.num_synonyms_per_word = num_synonyms_per_word
        self.spacy_models = {}
        self.spacy_models["multi"] = spacy.load("xx_ent_wiki_sm")
        self.spacy_models["en"] = spacy.load("en_core_web_sm")
        # TODO: Classify automatically
        self.economy_domains = ['finance', 'banking']

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


    def enhance_query(query, lang=None):
        if lang == None:
            lang = detect(query)
        
        if lang in ["de", "en"]:
            try:
                keyword_model = self.spacy_models[lang]
                keyword_model.add_pipe(WordnetAnnotator(keyword_model.lang), after='tagger')
            except:
                try:
                    self.spacy_models[lang] = spacy.load(lang + "_core_news_sm")
                    keyword_model = self.spacy_models[lang]
                except:
                    print('Please install the "' + lang + '_core_news_sm" spacy model')
        else:
            print("The language " + lang + " is actually not supported")
        
        enhanced_query = query
        enriched_sentence = []
        # For each token in the sentence
        for token in sentence:
            # We get those synsets within the desired domains
            synsets = token._.wordnet.wordnet_synsets_for_domain(self.economy_domains)
            if not synsets:
                enriched_sentence.append(token.text)
            else:
                lemmas_for_synset = [lemma for s in synsets for lemma in s.lemma_names()]
                # If we found a synset in the economy domains
                # we get the variants and add them to the enriched sentence
                enriched_sentence.append('{}'.format(' '.join(set(lemmas_for_synset))))
        return ' '.join(enriched_sentence)


        def convert_query(query, lang = None):
            new_query = self.enhance_query(query, lang)

            new_query = self.remove_stop(query, lang)

            return new_query
