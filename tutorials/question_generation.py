from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from haystack.question_generator import QuestionGenerator

text = "generate questions: Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum \
and first released in 1991, Python's design philosophy emphasizes code \
readability with its notable use of significant whitespace."

question_generator = QuestionGenerator(max_length=512, early_stopping=False)
print(question_generator.generate(text))