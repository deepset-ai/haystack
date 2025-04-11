from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.chat_message import ImageContent

### Simple python example

message = ChatMessage.from_user(
    content_parts=[
        "Describe this image in 20 words or less.",
        ImageContent.from_url(
            url="https://haystack.deepset.ai/images/model_providers_hu0cc9bf8102b4d45e75d89acd8a0ddbf4_163613_800x0_resize_q80_box_3.png"
        ),
    ]
)

generator = OpenAIChatGenerator(model="gpt-4o-mini")

response = generator.run(messages=[message])

print(response["replies"][0].text)
# The image features various AI platforms connected to a central symbol, symbolizing collaboration in artificial
# intelligence development.
