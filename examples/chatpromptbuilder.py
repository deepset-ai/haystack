import base64

from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.dataclasses.chat_message import ImageContent

### Use multimodal inputs with ChatPromptBuilder

template = """
{% message role="system" %}
    You are a helpful and joking assistant.
{% endmessage %}

{% message role="user" %}
    What's the difference between the two images?
    {% for image in images %}
        {{ image | for_template }}
    {% endfor %}
{% endmessage %}
"""


builder = ChatPromptBuilder(template=template)

paths = ["./test/test_files/images/apple.jpg", "./test/test_files/images/haystack-logo.png"]

image_contents = []
for path in paths:
    with open(path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    image_contents.append(ImageContent(base64_image=base64_image, detail="low"))

prompt = builder.run(template_variables={"images": image_contents})["prompt"]

generator = OpenAIChatGenerator(model="gpt-4o-mini")

response = generator.run(messages=prompt)

print(response["replies"][0].text)
# The first image shows a red apple, while the second image displays a logo for a software product called "Haystack"
# by DeepSet. The key difference is that one is a tangible fruit and the other is a visual representation of a company
# or product. Apples are for eating, and logos are for branding—one fills your stomach, and the other fills your screen!
# 🍏🖥️
