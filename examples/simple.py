from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.chat_message import ImageContent

### Simple python example

image_path = "./test/test_files/images/apple.jpg"
image_content = ImageContent.from_file_path(image_path, detail="low")

message = ChatMessage.from_user(content_parts=["Describe this image in 20 words or less.", image_content])

generator = OpenAIChatGenerator(model="gpt-4o-mini")

response = generator.run(messages=[message])

print(response["replies"][0].text)
# A fresh, ripe apple featuring red and green hues, resting on a bed of straw with a small leaf attached.
