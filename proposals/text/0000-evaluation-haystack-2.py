# mypy: ignore-errors
# pylint: skip-file
###
### This is an example script of how to use the eval function to evaluate a RAG Pipeline.
### For more information see the relative proposal.
###

import os

from haystack.preview import Pipeline
from haystack.preview.dataclasses.document import Document
from haystack.preview.components.retrievers.memory import MemoryBM25Retriever
from haystack.preview.document_stores.memory import MemoryDocumentStore
from haystack.preview.components.generators.openai.gpt35 import GPT35Generator
from haystack.preview.components.builders.prompt_builder import PromptBuilder


docstore = MemoryDocumentStore()

# Write some fake documents
docstore.write_documents(
    [
        Document(text="This is not the answer you are looking for.", metadata={"name": "Obi-Wan Kenobi"}),
        Document(text="This is the way.", metadata={"name": "Mandalorian"}),
        Document(text="The answer to life, the universe and everything is 42.", metadata={"name": "Deep Thought"}),
        Document(text="When you play the game of thrones, you win or you die.", metadata={"name": "Cersei Lannister"}),
        Document(text="Winter is coming.", metadata={"name": "Ned Stark"}),
    ]
)

# Create our retriever, we set top_k to 3 to get only the best 3 documents otherwise by default we get 10
retriever = MemoryBM25Retriever(document_store=docstore, top_k=3)

# Create our prompt template
template = """Given the context please answer the question.
Context:
{# We're receiving a list of lists, so we handle it like this #}
{% for list in documents %}
    {% for doc in list %}
        {{- doc -}};
    {% endfor %}
{% endfor %}
Question: {{ question }};
Answer:
"""
prompt_builder = PromptBuilder(template)

# We're using OpenAI gpt-3.5
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
generator = GPT35Generator(api_key=OPENAI_API_KEY)

# Build the pipeline
pipe = Pipeline()

pipe.add_component("docs_retriever", retriever)
pipe.add_component("builder", prompt_builder)
pipe.add_component("gpt35", generator)

pipe.connect("docs_retriever.documents", "builder.documents")
pipe.connect("builder.prompt", "gpt35.prompt")

# Run the pipeline
query = "What is the answer to life, the universe and everything?"
result = pipe.run({"docs_retriever": {"queries": [query]}, "builder": {"question": query}})

print(result["gpt35"]["replies"])


# These are the input that will be passed to the Pipeline when running eval, much like we've done a couple of lines above
inputs = [
    {"docs_retriever": {"queries": ["What is the answer?"]}, "builder": {"question": "What is the answer?"}},
    {
        "docs_retriever": {"queries": ["Take a deep breath and think. What is the answer?"]},
        "builder": {"question": "Take a deep breath and think. What is the answer?"},
    },
    {
        "docs_retriever": {"queries": ["What is the answer to life, the universe and everything?"]},
        "builder": {"question": "What is the answer to life, the universe and everything?"},
    },
]

# These are the expected output that will be compared to the actual output of the Pipeline.
# We have a dictionary for each input so that len(inputs) == len(expected_output).
# This gives the possibility to have different expected output for each different input.
# NOTE: I omitted the gpt35 metadata output because it's too long.
expected_output = [
    {
        # This is the output that we expect from the docs_retriever component
        "docs_retriever": {
            "documents": [
                [
                    Document(
                        text="The answer to life, the universe and everything is 42.", metadata={"name": "Deep Thought"}
                    ),
                    Document(text="This is not the answer you are looking for.", metadata={"name": "Obi-Wan Kenobi"}),
                    Document(text="This is the way.", metadata={"name": "Mandalorian"}),
                ]
            ]
        },
        # This is the output that we expect from the builder component
        "builder": {"prompt": "I should write the actual template here but I'm lazy so I won't."},
        # This is the output that we expect from the gpt35 component
        "gpt35": {"replies": ["The answer to life, the universe and everything is 42."], "metadata": {}},
    },
    {
        "docs_retriever": {
            "documents": [
                [
                    Document(
                        text="The answer to life, the universe and everything is 42.", metadata={"name": "Deep Thought"}
                    ),
                    Document(text="This is not the answer you are looking for.", metadata={"name": "Obi-Wan Kenobi"}),
                    Document(text="This is the way.", metadata={"name": "Mandalorian"}),
                ]
            ]
        },
        "builder": {"prompt": "I should write the actual template here but I'm lazy so I won't."},
        "gpt35": {"replies": ["The answer to life, the universe and everything is 42."], "metadata": {}},
    },
    {
        "docs_retriever": {
            "documents": [
                [
                    Document(
                        text="The answer to life, the universe and everything is 42.", metadata={"name": "Deep Thought"}
                    ),
                    Document(text="This is not the answer you are looking for.", metadata={"name": "Obi-Wan Kenobi"}),
                    Document(text="This is the way.", metadata={"name": "Mandalorian"}),
                ]
            ]
        },
        "builder": {"prompt": "I should write the actual template here but I'm lazy so I won't."},
        "gpt35": {"replies": ["The answer to life, the universe and everything is 42."], "metadata": {}},
    },
]

eval_result = eval(pipe, inputs=inputs, expected_output=expected_output)
metrics = result.calculate_metrics(Metric.SAS)
metrics.save("path/to/file.csv")
