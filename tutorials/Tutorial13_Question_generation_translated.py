from tqdm import tqdm
from pprint import pprint
from haystack.nodes import QuestionGenerator, FARMReader, TransformersTranslator
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.pipelines import QuestionAnswerGenerationPipeline, TranslationWrapperPipeline
from haystack.utils import launch_es, print_questions

""" 
This is a bare bones tutorial showing what is possible with the QuestionGenerator Node which automatically generates 
questions which the model thinks can be answered by a given document. 
"""


def tutorial13_question_generation():
    # Start Elasticsearch service via Docker
    launch_es()

    # text1 = "Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum and first released in 1991, Python's design philosophy emphasizes code readability with its notable use of significant whitespace."
    text1 = "Python ist eine interpretierte Hochsprachenprogrammiersprache für allgemeine Zwecke. Sie wurde von Guido van Rossum entwickelt und 1991 erstmals veröffentlicht. Die Design-Philosophie von Python legt den Schwerpunkt auf die Lesbarkeit des Codes und die Verwendung von viel Leerraum (Whitespace)."
    # text2 = "Princess Arya Stark is the third child and second daughter of Lord Eddard Stark and his wife, Lady Catelyn Stark. She is the sister of the incumbent Westerosi monarchs, Sansa, Queen in the North, and Brandon, King of the Andals and the First Men. After narrowly escaping the persecution of House Stark by House Lannister, Arya is trained as a Faceless Man at the House of Black and White in Braavos, using her abilities to avenge her family. Upon her return to Westeros, she exacts retribution for the Red Wedding by exterminating the Frey male line."
    text2 = "Prinzessin Arya Stark ist das dritte Kind und die zweite Tochter von Lord Eddard Stark und seiner Frau, Lady Catelyn Stark. Sie ist die Schwester der amtierenden Monarchen von Westerosi, Sansa, Königin im Norden, und Brandon, König der Andalen und der Ersten Menschen. Nachdem sie der Verfolgung des Hauses Stark durch das Haus Lannister nur knapp entkommen ist, wird Arya im Haus von Schwarz und Weiß in Braavos zum gesichtslosen Mann ausgebildet und nutzt ihre Fähigkeiten, um ihre Familie zu rächen. Nach ihrer Rückkehr nach Westeros übt sie Vergeltung für die Rote Hochzeit, indem sie die männliche Linie der Freys ausrottet."
    # text3 = "Dry Cleaning are an English post-punk band who formed in South London in 2018.[3] The band is composed of vocalist Florence Shaw, guitarist Tom Dowse, bassist Lewis Maynard and drummer Nick Buxton. They are noted for their use of spoken word primarily in lieu of sung vocals, as well as their unconventional lyrics. Their musical stylings have been compared to Wire, Magazine and Joy Division.[4] The band released their debut single, 'Magic of Meghan' in 2019. Shaw wrote the song after going through a break-up and moving out of her former partner's apartment the same day that Meghan Markle and Prince Harry announced they were engaged.[5] This was followed by the release of two EPs that year: Sweet Princess in August and Boundary Road Snacks and Drinks in October. The band were included as part of the NME 100 of 2020,[6] as well as DIY magazine's Class of 2020.[7] The band signed to 4AD in late 2020 and shared a new single, 'Scratchcard Lanyard'.[8] In February 2021, the band shared details of their debut studio album, New Long Leg. They also shared the single 'Strong Feelings'.[9] The album, which was produced by John Parish, was released on 2 April 2021.[10]"
    text3 = "Dry Cleaning ist eine englische Post-Punk-Band, die 2018 im Süden Londons gegründet wurde.[3] Die Band besteht aus Sängerin Florence Shaw, Gitarrist Tom Dowse, Bassist Lewis Maynard und Schlagzeuger Nick Buxton. Sie sind bekannt für ihre Verwendung von gesprochenem Wort anstelle von gesungenem Gesang sowie für ihre unkonventionellen Texte. Ihr musikalischer Stil wurde mit Wire, Magazine und Joy Division verglichen.[4] Die Band veröffentlichte 2019 ihre Debütsingle \"Magic of Meghan\". Shaw schrieb den Song nach einer Trennung und dem Auszug aus der Wohnung ihres ehemaligen Partners am selben Tag, an dem Meghan Markle und Prinz Harry ihre Verlobung bekannt gaben.[5] Im selben Jahr wurden zwei EPs veröffentlicht: Sweet Princess im August und Boundary Road Snacks and Drinks im Oktober. Die Band wurde in die NME 100 of 2020[6] und in die Class of 2020 des DIY Magazins aufgenommen.[7] Ende 2020 unterschrieb die Band bei 4AD und veröffentlichte eine neue Single, \"Scratchcard Lanyard\"[8] Im Februar 2021 gab die Band Details zu ihrem Debütalbum \"New Long Leg\" bekannt. Sie veröffentlichten auch die Single \"Strong Feelings\"[9] Das von John Parish produzierte Album wurde am 2. April 2021 veröffentlicht[10]."

    docs = [{"content": text1},
            {"content": text2},
            {"content": text3}]

    # Initialize document store and write in the documents
    document_store = ElasticsearchDocumentStore()
    document_store.write_documents(docs)


    """
    This pipeline takes a document as input, generates questions on it, and attempts to answer these questions using a 
    Reader model. In addition, let's use a translator to generate questions for documents in a different language where 
    no powerful generator model is available yet.
    """

    print("QuestionAnswerGenerationPipeline")
    print("===============================")

    question_generator = QuestionGenerator()
    reader = FARMReader("deepset/roberta-base-squad2")
    qag_pipeline = QuestionAnswerGenerationPipeline(question_generator, reader)

    in_translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-de-en")
    out_translator = TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-en-de")

    pipeline_with_translation = TranslationWrapperPipeline(input_translator=in_translator,
                                                           output_translator=out_translator,
                                                           pipeline=qag_pipeline)

    for idx, document in enumerate(tqdm(document_store)):
        print(f"\n * Generating questions and answers for document {idx}: {document.content[:100]}...\n")
        result = pipeline_with_translation.run(documents=[document])
        print_questions(result)
        #pprint(result)


if __name__ == "__main__":
    tutorial13_question_generation()

# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/
