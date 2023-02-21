from haystack.utils.import_utils import safe_import

from haystack.nodes.base import BaseComponent

from haystack.nodes.answer_generator import BaseGenerator, RAGenerator, Seq2SeqGenerator, OpenAIAnswerGenerator
from haystack.nodes.document_classifier import BaseDocumentClassifier, TransformersDocumentClassifier
from haystack.nodes.evaluator import EvalDocuments, EvalAnswers
from haystack.nodes.extractor import EntityExtractor, simplify_ner_for_qa
from haystack.nodes.file_classifier import FileTypeClassifier
from haystack.nodes.file_converter import (
    BaseConverter,
    DocxToTextConverter,
    ImageToTextConverter,
    MarkdownConverter,
    PDFToTextConverter,
    PDFToTextOCRConverter,
    TikaConverter,
    TikaXHTMLParser,
    TextConverter,
    AzureConverter,
    ParsrConverter,
    CsvTextConverter,
    JsonConverter,
)
from haystack.nodes.image_to_text import TransformersImageToText
from haystack.nodes.label_generator import PseudoLabelGenerator
from haystack.nodes.other import Docs2Answers, JoinDocuments, RouteDocuments, JoinAnswers, DocumentMerger, Shaper
from haystack.nodes.preprocessor import BasePreProcessor, PreProcessor
from haystack.nodes.prompt import PromptNode, PromptTemplate, PromptModel
from haystack.nodes.query_classifier import SklearnQueryClassifier, TransformersQueryClassifier
from haystack.nodes.question_generator import QuestionGenerator
from haystack.nodes.ranker import BaseRanker, SentenceTransformersRanker
from haystack.nodes.reader import BaseReader, FARMReader, TransformersReader, TableReader, RCIReader
from haystack.nodes.retriever import (
    BaseRetriever,
    DenseRetriever,
    DensePassageRetriever,
    EmbeddingRetriever,
    BM25Retriever,
    ElasticsearchRetriever,
    FilterRetriever,
    MultihopEmbeddingRetriever,
    ElasticsearchFilterOnlyRetriever,
    TfidfRetriever,
    Text2SparqlRetriever,
    TableTextRetriever,
    MultiModalRetriever,
)
from haystack.nodes.summarizer import BaseSummarizer, TransformersSummarizer
from haystack.nodes.translator import BaseTranslator, TransformersTranslator

Crawler = safe_import("haystack.nodes.connector.crawler", "Crawler", "crawler")  # Has optional dependencies
AnswerToSpeech = safe_import(
    "haystack.nodes.audio.answer_to_speech", "AnswerToSpeech", "audio"
)  # Has optional dependencies
DocumentToSpeech = safe_import(
    "haystack.nodes.audio.document_to_speech", "DocumentToSpeech", "audio"
)  # Has optional dependencies
