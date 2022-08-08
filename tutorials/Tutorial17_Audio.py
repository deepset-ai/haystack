# ## Make Your QA Pipelines Talk!
#
# Question answering works primarily on text, but Haystack provides some features for
# audio files that contain speech as well.
#
# In this tutorial, we're going to see how to use `AnswerToSpeech` to convert answers
# into audio files.
#
import logging

# We configure how logging messages should be displayed and which log level should be used before importing Haystack.
# Example log message:
# INFO - haystack.utils.preprocessing -  Converting data/tutorial1/218_Olenna_Tyrell.txt
# Default log level in basicConfig is WARNING so the explicit parameter is not necessary but can be changed easily:
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.utils import fetch_archive_from_http, launch_es, print_answers
from haystack.nodes import FARMReader, BM25Retriever
from pprint import pprint
from pathlib import Path
from haystack import Pipeline
from haystack.nodes import FileTypeClassifier, TextConverter, PreProcessor, AnswerToSpeech, DocumentToSpeech


def tutorial17_audio_features():

    ############################################################################################
    #
    # ## Part 1: INDEXING
    #
    # First of all, we create a pipeline that populates the document store. See Tutorial 1 for more details about these steps.
    #
    # To the basic version, we can add here a DocumentToSpeech node that also generates an audio file for each of the
    # indexed documents. During querying, this will make it easier, to access the audio version of the documents the answers
    # were extracted from.

    # Connect to Elasticsearch
    launch_es(sleep=30)
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

    # Get the documents
    documents_path = "data/tutorial17"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt17.zip"
    fetch_archive_from_http(url=s3_url, output_dir=documents_path)

    # List all the paths
    file_paths = [p for p in Path(documents_path).glob("**/*")]

    # Note: In this example, we're going to use only one text file from the wiki, as the DocumentToSpeech node is relatively slow
    # on CPU machines. Comment out this line to use all documents from the dataset if you machine is powerful enough.
    file_paths = [p for p in file_paths if "Stormborn" in p.name]

    # Prepare some basic metadata for the files
    files_metadata = [{"name": path.name} for path in file_paths]

    # Here we create a basic indexing pipeline
    indexing_pipeline = Pipeline()

    # - Makes sure the file is a TXT file (FileTypeClassifier node)
    classifier = FileTypeClassifier()
    indexing_pipeline.add_node(classifier, name="classifier", inputs=["File"])

    # - Converts a file into text and performs basic cleaning (TextConverter node)
    text_converter = TextConverter(remove_numeric_tables=True)
    indexing_pipeline.add_node(text_converter, name="text_converter", inputs=["classifier.output_1"])

    # - Pre-processes the text by performing splits and adding metadata to the text (PreProcessor node)
    preprocessor = PreProcessor(
        clean_whitespace=True,
        clean_empty_lines=True,
        split_length=100,
        split_overlap=50,
        split_respect_sentence_boundary=True,
    )
    indexing_pipeline.add_node(preprocessor, name="preprocessor", inputs=["text_converter"])

    #
    # DocumentToSpeech
    #
    # Here is where we convert all documents to be indexed into SpeechDocuments, that will hold not only
    # the text content, but also their audio version.
    doc2speech = DocumentToSpeech(
        model_name_or_path="espnet/kan-bayashi_ljspeech_vits", generated_audio_dir=Path("./generated_audio_documents")
    )
    indexing_pipeline.add_node(doc2speech, name="doc2speech", inputs=["preprocessor"])

    # - Writes the resulting documents into the document store (ElasticsearchDocumentStore node from the previous cell)
    indexing_pipeline.add_node(document_store, name="document_store", inputs=["doc2speech"])

    # Then we run it with the documents and their metadata as input
    indexing_pipeline.run(file_paths=file_paths, meta=files_metadata)

    # You can now check the document store and verify that documents have been enriched with a path
    # to the generated audio file
    document = next(document_store.get_all_documents_generator())
    pprint(document)

    # Sample output:
    #
    # <Document: {
    # 'content': "'Stormborn' received praise from critics, who considered Euron Greyjoy's raid on Yara's Iron Fleet,
    #             the assembly of Daenerys' allies at Dragonstone, and Arya's reunion with her direwolf Nymeria as
    #             highlights of the episode. In the United States, it achieved a viewership of 9.27 million in its
    #             initial broadcast.",
    # 'content_type': 'audio',
    # 'score': None,
    # 'meta': {
    #       'content_audio': './generated_audio_documents/f218707624d9c4f9487f508e4603bf5b.wav',
    #       '__initialised__': True,
    #       'type': 'generative',
    #       '_split_id': 0,
    #       'audio_format': 'wav',
    #       'sample_rate': 22050,
    #       'name': '2_Stormborn.txt'},
    #       'embedding': None,
    #       'id': '2733e698301f8f94eb70430b874177fd'
    # }>

    ############################################################################################
    #
    # ## Part 2: QUERYING
    #
    # Now we will create a pipeline very similar to the basic ExtractiveQAPipeline of Tutorial 1,
    # with the addition of a node that converts our answers into audio files!

    retriever = BM25Retriever(document_store=document_store)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2-distilled", use_gpu=True)
    answer2speech = AnswerToSpeech(
        model_name_or_path="espnet/kan-bayashi_ljspeech_vits",
        generated_audio_dir=Path(__file__).parent / "audio_answers",
    )

    audio_pipeline = Pipeline()
    audio_pipeline.add_node(retriever, name="Retriever", inputs=["Query"])
    audio_pipeline.add_node(reader, name="Reader", inputs=["Retriever"])
    audio_pipeline.add_node(answer2speech, name="AnswerToSpeech", inputs=["Reader"])

    prediction = audio_pipeline.run(
        query="Who was born during a storm?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
    )

    # Now you can either print the object directly
    print("\n\nRaw object:\n")
    pprint(prediction)

    # Sample output:
    # {
    #     'answers': [ <SpeechAnswer:
    #                       answer_audio=PosixPath('generated_audio_answers/fc704210136643b833515ba628eb4b2a.wav'),
    #                       answer="Daenerys Targaryen",
    #                       context_audio=PosixPath('generated_audio_answers/8c562ebd7e7f41e1f9208384957df173.wav'),
    #                       context='...'
    #                       type='extractive', score=0.9919578731060028,
    #                       offsets_in_document=[{'start': 608, 'end': 615}], offsets_in_context=[{'start': 72, 'end': 79}],
    #                       document_id='cc75f739897ecbf8c14657b13dda890e', meta={'name': '43_Arya_Stark.txt'}}  >,
    #                  <SpeechAnswer:
    #                       answer_audio=PosixPath('generated_audio_answers/07d6265486b22356362387c5a098ba7d.wav'),
    #                       answer="Daenerys",
    #                       context_audio=PosixPath('generated_audio_answers/3f1ca228d6c4cfb633e55f89e97de7ac.wav'),
    #                       context='...'
    #                       type='extractive', score=0.9767240881919861,
    #                       offsets_in_document=[{'start': 3687, 'end': 3801}], offsets_in_context=[{'start': 18, 'end': 132}],
    #                       document_id='9acf17ec9083c4022f69eb4a37187080', meta={'name': '43_Arya_Stark.txt'}}>,
    #                  ...
    #                ]
    #     'documents': [ <SpeechDocument:
    #                        content_type='text', score=0.8034909798951382, meta={'name': '43_Arya_Stark.txt'}, embedding=None, id=d1f36ec7170e4c46cde65787fe125dfe',
    #                        content_audio=PosixPath('generated_audio_documents/07d6265486b22356362387c5a098ba7d.wav'),
    #                        content='The title of the episode refers to both Daenerys Targaryen, who was born during a  ...'>,
    #                    <SpeechDocument:
    #                        content_type='text', score=0.8002150354529785, meta={'name': '191_Gendry.txt'}, embedding=None, id='dd4e070a22896afa81748d6510006d2',
    #                        content_audio=PosixPath('generated_audio_documents/07d6265486b22356362387c5a098ba7d.wav'),
    #                        content='"Stormborn" received praise from critics, who considered Euron Greyjoy's raid on ...'>,
    #                    ...
    #                  ],
    #     'no_ans_gap':  11.688868522644043,
    #     'node_id': 'Reader',
    #     'params': {'Reader': {'top_k': 5}, 'Retriever': {'top_k': 5}},
    #     'query': 'Who was born during a storm?',
    #     'root_node': 'Query'
    # }

    # Or use a util to simplify the output
    # Change `minimum` to `medium` or `all` to raise the level of detail
    print("\n\nSimplified output:\n")
    print_answers(prediction, details="minimum")

    # Sample output:
    #
    # Query: Who was born during a storm
    # Answers:
    # [   {   'answer_audio': PosixPath('generated_audio_answers/07d6265486b22356362387c5a098ba7d.wav'),
    #         'answer': 'Daenerys Targaryen',
    #         'context_transcript': PosixPath('generated_audio_answers/3f1ca228d6c4cfb633e55f89e97de7ac.wav'),
    #         'context': ' refers to both Daenerys Targaryen, who was born during a terrible storm, and '},
    #    {   'answer_audio': PosixPath('generated_audio_answers/83c3a02141cac4caffe0718cfd6c405c.wav'),
    #        'answer': 'Daenerys',
    #        'context_audio': PosixPath('generated_audio_answers/8c562ebd7e7f41e1f9208384957df173.wav'),
    #        'context': 'The title of the episode refers to both Daenerys Targaryen, who was born during a terrible storm'},
    #    ...

    # The document the first answer was extracted from
    original_document = [doc for doc in prediction["documents"] if doc.id == prediction["answers"][0].document_id][0]
    pprint(original_document)

    # Sample output
    #
    # <Document: {
    #   'content': '"'''Stormborn'''" is the second episode of the seventh season of HBO's fantasy television
    #               series ''Game of Thrones'', and the 62nd overall. The episode was written by Bryan Cogman,
    #               and directed by Mark Mylod. The title of the episode refers to both Daenerys Targaryen,
    #               who was born during a terrible storm, and Euron Greyjoy, who declares himself to be "the storm".',
    #   'content_type': 'audio',
    #   'score': 0.6269117688771539,
    #   'embedding': None,
    #   'id': '9352f650b36f93ab99684fd4746af5c1'
    #   'meta': {
    #       'content_audio': '/home/sara/work/haystack/generated_audio_documents/2c9223d47801b0918f2db2ad778c3d5a.wav',
    #       'type': 'generative',
    #       '_split_id': 19,
    #       'audio_format': 'wav',
    #       'sample_rate': 22050,
    #       'name': '2_Stormborn.txt'}
    # }>


if __name__ == "__main__":
    tutorial17_audio_features()


# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/
