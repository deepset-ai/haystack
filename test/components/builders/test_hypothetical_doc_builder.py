from typing import List

import pytest

from haystack.builders import HypotheticalDocumentEmbedder


def test_from_dict():
    data = {
        "type": "haystack.components.builders.hypothetical_doc_query.HypotheticalDocumentEmbedder",
        "pipeline": {
            "metadata": {},
            "max_loops_allowed": 100,
            "components": {
                "prompt_builder": {
                    "type": "haystack.components.builders.prompt_builder.PromptBuilder",
                    "init_parameters": {
                        "template": "\nGiven a question, generate a paragraph of text that answers the question.\nQuestion: {{question}}\nParagraph:\n"
                    },
                },
                "generator": {
                    "type": "haystack.components.generators.openai.OpenAIGenerator",
                    "init_parameters": {
                        "model": "gpt-3.5-turbo",
                        "streaming_callback": None,
                        "api_base_url": None,
                        "generation_kwargs": {"n": 5, "temperature": 0.75, "max_tokens": 400},
                        "system_prompt": None,
                        "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True},
                    },
                },
            },
            "connections": [{"sender": "prompt_builder.prompt", "receiver": "generator.prompt"}],
        },
        "init_params": {"instruct_llm": "gpt-3.5-turbo", "nr_completions": 5},
    }

    hyde = HypotheticalDocumentEmbedder.from_dict(data)
    assert hyde.instruct_llm == "gpt-3.5-turbo"
    assert hyde.nr_completions == 5
    assert hyde.pipeline.to_dict() == data["pipeline"]
    assert (
        hyde.pipeline.components["prompt_builder"].template
        == "\nGiven a question, generate a paragraph of text that answers the question.\nQuestion: {{question}}\nParagraph:\n"
    )
    assert hyde.pipeline.components["generator"].model == "gpt-3.5-turbo"
    assert hyde.pipeline.components["generator"].generation_kwargs == {"n": 5, "temperature": 0.75, "max_tokens": 400}
