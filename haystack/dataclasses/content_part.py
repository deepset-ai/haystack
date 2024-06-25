from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union

from .byte_stream import ByteStream


class ContentType(str, Enum):
    """Enumeration representing the possible content types of a ChatMessage."""

    TEXT = "text"
    IMAGE_URL = "image_url"
    IMAGE_BASE64 = "image_base64"

    def to_openai_format(self) -> str:
        """
        Returns the string that OpenAI expects inside the 'type' key on the dictionary.

        :returns: The string expected by OpenAI.
        """

        return {ContentType.TEXT: "text", ContentType.IMAGE_URL: "image_url", ContentType.IMAGE_BASE64: "image_url"}[
            self
        ]


class ImageDetail(str, Enum):
    """Enumeration representing the possible image quality options in OpenAI Chat API."""

    HIGH = "high"
    LOW = "low"
    AUTO = "auto"


@dataclass
class ContentPart:
    """
    Represents a part of the content on a message in a LLM chat conversation.

    :param type: The content type of the message, weather it is regular text, an image url, ...
    :param content: The actual content of this part of the message.
    """

    type: ContentType
    content: Union[str, ByteStream]
    image_detail: Optional[ImageDetail] = None

    def to_openai_format(self) -> Dict[str, Any]:
        """
        Convert the content part to the format expected by OpenAI's Chat API.

        See the [API reference](https://platform.openai.com/docs/api-reference/chat/create) for details.

        :returns: A dictionary representation of the content part.
        """

        content = {"type": self.type.to_openai_format()}

        if self.type is ContentType.TEXT:
            content["text"] = self.content

        elif self.type is ContentType.IMAGE_URL:
            content["image_url"] = {"url": self.content}

            if self.image_detail is not None:
                content["image_url"]["detail"] = self.image_detail.value

        elif self.type is ContentType.IMAGE_BASE64:
            content["image_url"] = {  # TODO: check if png images work as well
                "url": f"data:image/jpeg;base64,{self.content.to_string()}"
            }

            if self.image_detail is not None:
                content["image_url"]["detail"] = self.image_detail.value

        else:
            ValueError("Invalid content type, only ByteStream and string are supported.")

        return content

    @classmethod
    def from_text(cls, text: str) -> "ContentPart":
        """
        Create a ContentPart from a string of text.

        :param text: The text content.
        :returns: A new ContentPart instance.
        """
        return cls(type=ContentType.TEXT, content=text)

    @classmethod
    def from_image_url(cls, url: str, image_detail: Optional[ImageDetail] = None) -> "ContentPart":
        """
        Create a ContentPart from an image url.

        :param text: The url of the image.
        :returns: A new ContentPart instance.
        """
        return cls(type=ContentType.IMAGE_URL, content=url, image_detail=image_detail)

    @classmethod
    def from_base64_image(cls, image: ByteStream, image_detail: Optional[ImageDetail] = None) -> "ContentPart":
        """
        Create a ContentPart from a base64 encoded image.

        :param text: The base64 byte representation of the image.
        :returns: A new ContentPart instance.
        """
        # TODO: add support for input str?
        return cls(type=ContentType.IMAGE_BASE64, content=image, image_detail=image_detail)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts ContentPart into a dictionary.

        :returns:
            Serialized version of the object.
        """
        data = asdict(self)
        data["type"] = self.type.value

        # Only add image_detail if it is not None
        image_detail = data.pop("image_detail")
        if image_detail is not None:
            data["image_detail"] = image_detail.value

        if isinstance(self.content, ByteStream):
            data["content"] = self.content.to_string()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContentPart":
        """
        Creates a new ContentPart object from a dictionary.

        :param data:
            The dictionary to build the ContentPart object.
        :returns:
            The created object.
        """
        data["type"] = ContentType(data["type"])
        if "image_detail" in data:
            data["image_detail"] = ImageDetail(data["image_detail"])

        if data["type"] is ContentType.IMAGE_BASE64:  # Convert to bytes
            data["content"] = ByteStream.from_string(data["content"])

        return cls(**data)
