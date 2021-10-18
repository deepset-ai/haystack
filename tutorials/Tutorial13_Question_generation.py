from haystack.question_generator import QuestionGenerator
from haystack.utils import launch_es
from haystack.document_store import ElasticsearchDocumentStore
from haystack.retriever import ElasticsearchRetriever
from pprint import pprint
from haystack.reader import FARMReader
from tqdm import tqdm
from haystack.pipeline import QuestionGenerationPipeline, RetrieverQuestionGenerationPipeline, QuestionAnswerGenerationPipeline

""" 
This is a bare bones tutorial showing what is possible with the QuestionGenerator Node which automatically generates 
questions which the model thinks can be answered by a given document. 
"""

# Start Elasticsearch service via Docker
launch_es()

text1 = "Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum and first released in 1991, Python's design philosophy emphasizes code readability with its notable use of significant whitespace."
text2 = "Princess Arya Stark is the third child and second daughter of Lord Eddard Stark and his wife, Lady Catelyn Stark. She is the sister of the incumbent Westerosi monarchs, Sansa, Queen in the North, and Brandon, King of the Andals and the First Men. After narrowly escaping the persecution of House Stark by House Lannister, Arya is trained as a Faceless Man at the House of Black and White in Braavos, using her abilities to avenge her family. Upon her return to Westeros, she exacts retribution for the Red Wedding by exterminating the Frey male line."
text3 = "Dry Cleaning are an English post-punk band who formed in South London in 2018.[3] The band is composed of vocalist Florence Shaw, guitarist Tom Dowse, bassist Lewis Maynard and drummer Nick Buxton. They are noted for their use of spoken word primarily in lieu of sung vocals, as well as their unconventional lyrics. Their musical stylings have been compared to Wire, Magazine and Joy Division.[4] The band released their debut single, 'Magic of Meghan' in 2019. Shaw wrote the song after going through a break-up and moving out of her former partner's apartment the same day that Meghan Markle and Prince Harry announced they were engaged.[5] This was followed by the release of two EPs that year: Sweet Princess in August and Boundary Road Snacks and Drinks in October. The band were included as part of the NME 100 of 2020,[6] as well as DIY magazine's Class of 2020.[7] The band signed to 4AD in late 2020 and shared a new single, 'Scratchcard Lanyard'.[8] In February 2021, the band shared details of their debut studio album, New Long Leg. They also shared the single 'Strong Feelings'.[9] The album, which was produced by John Parish, was released on 2 April 2021.[10]"

docs = [{"content": text1},
        {"content": text2},
        {"content": text3}]

# Initialize document store and write in the documents
document_store = ElasticsearchDocumentStore()
document_store.write_documents(docs)

# Initialize Question Generator
question_generator = QuestionGenerator()

"""
The most basic version of a question generator pipeline takes a document as input and outputs generated questions
which the the document can answer.
"""

# QuestionGenerationPipeline
question_generation_pipeline = QuestionGenerationPipeline(question_generator)
for document in document_store:
        result = question_generation_pipeline.run(documents=[document])
        pprint(result)

"""
This pipeline takes a query as input. It retrievers relevant documents and then generates questions based on these.
"""

# RetrieverQuestionGenerationPipeline
retriever = ElasticsearchRetriever(document_store=document_store)
rqg_pipeline = RetrieverQuestionGenerationPipeline(retriever, question_generator)
result = rqg_pipeline.run(query="Arya Stark")
pprint(result)

"""
This pipeline takes a document as input, generates questions on it, and attempts to answer these questions using
a Reader model
"""

# QuestionAnswerGenerationPipeline
reader = FARMReader("deepset/roberta-base-squad2")
qag_pipeline = QuestionAnswerGenerationPipeline(question_generator, reader)
for document in tqdm(document_store):
    result = qag_pipeline.run(documents=[document])
    pprint(result)

# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/