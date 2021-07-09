from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from haystack.question_generator import QuestionGenerator
from haystack.utils import launch_es
from haystack.document_store import ElasticsearchDocumentStore
from haystack.retriever import ElasticsearchRetriever
from pprint import pprint
from haystack import Pipeline
from haystack.reader import FARMReader
from tqdm import tqdm
from haystack import Document
from haystack.pipeline import QuestionGenerationPipeline, RetrieverQuestionGenerationPipeline, QuestionAnswerGenerationPipeline

launch_es()

text = "Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum \
and first released in 1991, Python's design philosophy emphasizes code \
readability with its notable use of significant whitespace."

text2 = "Princess Arya Stark is the third child and second daughter of Lord Eddard Stark and his wife, Lady Catelyn Stark. She is the sister of the incumbent Westerosi monarchs, Sansa, Queen in the North, and Brandon, King of the Andals and the First Men. After narrowly escaping the persecution of House Stark by House Lannister, Arya is trained as a Faceless Man at the House of Black and White in Braavos, using her abilities to avenge her family. Upon her return to Westeros, she exacts retribution for the Red Wedding by exterminating the Frey male line."

text3 = "Dry Cleaning are an English post-punk band who formed in South London in 2018.[3] The band is composed of vocalist Florence Shaw, guitarist Tom Dowse, bassist Lewis Maynard and drummer Nick Buxton. They are noted for their use of spoken word primarily in lieu of sung vocals, as well as their unconventional lyrics. Their musical stylings have been compared to Wire, Magazine and Joy Division.[4] The band released their debut single, 'Magic of Meghan' in 2019. Shaw wrote the song after going through a break-up and moving out of her former partner's apartment the same day that Meghan Markle and Prince Harry announced they were engaged.[5] This was followed by the release of two EPs that year: Sweet Princess in August and Boundary Road Snacks and Drinks in October. The band were included as part of the NME 100 of 2020,[6] as well as DIY magazine's Class of 2020.[7] The band signed to 4AD in late 2020 and shared a new single, 'Scratchcard Lanyard'.[8] In February 2021, the band shared details of their debut studio album, New Long Leg. They also shared the single 'Strong Feelings'.[9] The album, which was produced by John Parish, was released on 2 April 2021.[10]"

docs = [{"text": text},
        {"text": text2},
        {"text": text3}]
document_store = ElasticsearchDocumentStore()
document_store.write_documents(docs)

question_generator = QuestionGenerator()


# # Pipeline 1
# question_generation_pipeline = QuestionGenerationPipeline(question_generator)
# for document in document_store:
#         result = question_generation_pipeline.run(documents=[document])


# # Pipeline 2
# retriever = ElasticsearchRetriever(document_store=document_store)
# rqg_pipeline = RetrieverQuestionGenerationPipeline(retriever, question_generator)
# result = rqg_pipeline.run(query="Arya Stark")
# pprint(result)


# Pipeline 3
reader = FARMReader("deepset/roberta-base-squad2")
qag_pipeline = QuestionAnswerGenerationPipeline(question_generator, reader)
for document in tqdm(document_store):
    result = qag_pipeline.run(document=document)



# Pipeline3
reader = FARMReader("deepset/roberta-base-squad2")

results = []
for document in tqdm(document_store):
        questions = question_generator.generate(text=document.text)
        results.append({"generated_questions": questions,
                        "document_text": document.text,
                        "document_id": document.id})

ret = []
for result in tqdm(results):
    generated_questions = result["generated_questions"]
    document_text = result["document_text"]
    document_id = result["document_id"]
    for q in generated_questions:
        answers = reader.predict(query=q, documents=[Document.from_dict({"text": document_text})])["answers"]
        ret.append({"document_text": document_text,
                        "document_id": document_id,
                        "question": q,
                        "answers": answers})

curr_text = ""
for r in ret:
    if curr_text != r["document_text"]:
        curr_text = r["document_text"]
        print(f"Document: {curr_text}")
    print(f"\t{r['question']}")
    for a in r["answers"]:
        print(f"\t\t{a['answer']}")
        break
