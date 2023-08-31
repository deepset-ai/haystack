import os

from haystack.preview.components.generators.openai.chatgpt import ChatGPTGenerator

stream_response = False

llm = ChatGPTGenerator(
    api_key=os.environ.get("OPENAI_API_KEY"), model_name="gpt-3.5-turbo", max_tokens=256, stream=stream_response
)

responses = llm.run(prompts=["What is the meaning of life?"])
if not stream_response:
    print(responses)
