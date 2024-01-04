import json
import os

from haystack import Pipeline
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
import random
from haystack import component
from typing import Optional, List

import pydantic
from pydantic import BaseModel, ValidationError

import logging

logging.basicConfig()
logging.getLogger("haystack.core.pipeline.pipeline").setLevel(logging.DEBUG)


# Let's define a simple schema for the data we want to extract from a passsage via the LLM
# We want the output from our LLM to be always compliant with this
class City(BaseModel):
    name: str
    country: str
    population: int


class CitiesData(BaseModel):
    cities: List[City]


schema = CitiesData.schema_json(indent=2)


# We then create a simple, custom Haystack component that takes the LLM output
# and validates if this is compliant with our schema.
# If not, it returns also the error message so that we have a better chance of correcting it in the next loop
@component
class OutputParser:
    def __init__(self, pydantic_model: pydantic.BaseModel):
        self.pydantic_model = pydantic_model
        self.iteration_counter = 0

    @component.output_types(valid=List[str], invalid=Optional[List[str]], error_message=Optional[str])
    def run(self, replies: List[str]):
        self.iteration_counter += 1

        # let's simulate a corrupt JSON with 30% probability by adding extra brackets (for demo purposes)
        if random.randint(0, 100) < 30:
            replies[0] = "{{" + replies[0]

        try:
            output_dict = json.loads(replies[0])
            self.pydantic_model.parse_obj(output_dict)
            print(
                f"OutputParser at Iteration {self.iteration_counter}: Valid JSON from LLM - No need for looping: {replies[0]}"
            )
            return {"valid": replies}

        except (ValueError, ValidationError) as e:
            print(
                f"OutputParser at Iteration {self.iteration_counter}: Invalid JSON from LLM - Let's try again.\n"
                f"Output from LLM:\n {replies[0]} \n"
                f"Error from OutputParser: {e}"
            )
            return {"invalid": replies, "error_message": str(e)}


# Let's create a prompt that always includes the basic instructions for creating our JSON, and optionally, information from any previously failed attempt (corrupt JSON + error message from parsing it).
# The Jinja2 templating language gives us full flexibility here to adjust the prompt dynamically depending on which inputs are available
prompt_template = """
 Create a JSON object from the information present in this passage: {{passage}}.
 Only use information that is present in the passage. Follow this JSON schema, but only return the actual instances without any additional schema definition:"
 {{schema}}
 Make sure your response is a dict and not a list.
 {% if replies and error_message %}
    You already created the following output in a previous attempt: {{replies}}
    However, this doesn't comply with the format requirements from above and triggered this Python exception: {{ error_message}}
    Correct the output and try again. Just return the corrected output without any extra explanations.
  {% endif %}
"""

# Let's build the pipeline (Make sure to set OPENAI_API_KEY as an environment variable)
pipeline = Pipeline(max_loops_allowed=5)
pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
pipeline.add_component(instance=OpenAIGenerator(), name="llm")
pipeline.add_component(instance=OutputParser(pydantic_model=CitiesData), name="output_parser")  # type: ignore

pipeline.connect("prompt_builder", "llm")
pipeline.connect("llm", "output_parser")
pipeline.connect("output_parser.invalid", "prompt_builder.replies")
pipeline.connect("output_parser.error_message", "prompt_builder.error_message")

# Now, let's run our pipeline with an example passage that we want to convert into our JSON format
passage = "Berlin is the capital of Germany. It has a population of 3,850,809"
result = pipeline.run({"prompt_builder": {"passage": passage, "schema": schema}})

print(result)
