from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.chat_message import ImageContent, ChatMessage, TextContent, ChatRole
from haystack.components.generators.chat.openai import OpenAIChatGenerator
import base64


### Simple python example

message = ChatMessage.from_user(content_parts=[
    "Describe this image in 20 words or less.",
    ImageContent.from_url(url="https://haystack.deepset.ai/images/model_providers_hu0cc9bf8102b4d45e75d89acd8a0ddbf4_163613_800x0_resize_q80_box_3.png")
])

generator = OpenAIChatGenerator(model="gpt-4o-mini")

response = generator.run(messages=[message])

print(response["replies"][0].text)
# The image features various AI platforms connected to a central symbol, symbolizing collaboration in artificial 
# intelligence development.



### ChatPromptBuilder
template = [ChatMessage.from_user(content_parts=[
    "What's the difference between the two images? Be short and concise.",
    ImageContent(base64_image="{{base64_image}}", provider_options={"detail": "low"}),
    ImageContent(base64_image="{{base64_image2}}", provider_options={"detail": "low"})
])]



builder = ChatPromptBuilder(template=template)

with open("test/test_files/images/apple.jpg", "rb") as f:
    base64_image = base64.b64encode(f.read()).decode("utf-8")

with open("test/test_files/images/haystack-logo.png", "rb") as f:
    base64_image2 = base64.b64encode(f.read()).decode("utf-8")

prompt = builder.run(template_variables={"base64_image": base64_image, "base64_image2": base64_image2})["prompt"]

generator = OpenAIChatGenerator(model="gpt-4o-mini")

response = generator.run(messages=prompt)

print(response["replies"][0].text)
# The first image shows a red and green apple resting on straw, while the second image is a logo for "Haystack by 
# Deepset," featuring stylized typography and colors typical of branding.