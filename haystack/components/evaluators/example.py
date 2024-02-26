# A valid OpenAI API key must be provided as an environment variable "OPENAI_API_KEY" to run this example.
from typing import List

from haystack import Pipeline

from haystack.components.evaluators import LLMEvaluator

QUESTIONS = ["Which is the most popular global sport?", "Who created the Python language?"]
CONTEXTS = [
    [
        "The popularity of sports can be measured in various ways, including TV viewership, social media presence, number of participants, and economic impact. Football is undoubtedly the world's most popular sport with major events like the FIFA World Cup and sports personalities like Ronaldo and Messi, drawing a followership of more than 4 billion people."
    ],
    [
        "Python, created by Guido van Rossum in the late 1980s, is a high-level general-purpose programming language. Its design philosophy emphasizes code readability, and its language constructs aim to help programmers write clear, logical code for both small and large-scale software projects."
    ],
]
RESPONSES = [
    "Football is the most popular sport with around 4 billion followers worldwide",
    "Python language was created by Guido van Rossum.",
]

GROUND_TRUTHS = ["Football is the most popular sport", "Python language was created by Guido van Rossum."]

pipeline = Pipeline()
evaluator = LLMEvaluator(
    api="openai",
    inputs=[("responses", List[str])],
    outputs=["score"],
    instruction="Is this answer problematic for children?",
)
pipeline.add_component("evaluator", evaluator)

results = pipeline.run({"evaluator": {"responses": RESPONSES}})

for output in results["evaluator"]["results"]:
    print(output)
