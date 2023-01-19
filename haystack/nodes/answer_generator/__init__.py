from haystack.utils.import_utils import safe_import

from haystack.nodes.answer_generator.base import BaseGenerator
from haystack.nodes.answer_generator.transformers import RAGenerator, Seq2SeqGenerator

OpenAIAnswerGenerator = safe_import("haystack.nodes.answer_generator.openai", "OpenAIAnswerGenerator", "openai")
