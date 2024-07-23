from typing import Any, Dict, List, Optional
from jinja2 import TemplateSyntaxError
import pytest

from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack import component
from haystack.core.pipeline.pipeline import Pipeline
from haystack.dataclasses.chat_message import ChatMessage
from haystack.dataclasses.byte_stream import ByteStream
from haystack.dataclasses.document import Document


class TestChatPromptBuilder:
    def test_init(self):
        builder = ChatPromptBuilder(
            template=[
                ChatMessage.from_user(content="This is a {{ variable }}"),
                ChatMessage.from_system(content="This is a {{ variable2 }}"),
            ]
        )
        assert builder.required_variables == []
        assert builder.template[0].content == "This is a {{ variable }}"
        assert builder.template[1].content == "This is a {{ variable2 }}"
        assert builder._variables is None
        assert builder._required_variables is None

        # we have inputs that contain: template, template_variables + inferred variables
        inputs = builder.__haystack_input__._sockets_dict
        assert set(inputs.keys()) == {"template", "template_variables", "variable", "variable2"}
        assert inputs["template"].type == Optional[List[ChatMessage]]
        assert inputs["template_variables"].type == Optional[Dict[str, Any]]
        assert inputs["variable"].type == Any
        assert inputs["variable2"].type == Any

        # response is always prompt
        outputs = builder.__haystack_output__._sockets_dict
        assert set(outputs.keys()) == {"prompt"}
        assert outputs["prompt"].type == List[ChatMessage]

    def test_init_without_template(self):
        variables = ["var1", "var2"]
        builder = ChatPromptBuilder(variables=variables)
        assert builder.template is None
        assert builder.required_variables == []
        assert builder._variables == variables
        assert builder._required_variables is None

        # we have inputs that contain: template, template_variables + variables
        inputs = builder.__haystack_input__._sockets_dict
        assert set(inputs.keys()) == {"template", "template_variables", "var1", "var2"}
        assert inputs["template"].type == Optional[List[ChatMessage]]
        assert inputs["template_variables"].type == Optional[Dict[str, Any]]
        assert inputs["var1"].type == Any
        assert inputs["var2"].type == Any

        # response is always prompt
        outputs = builder.__haystack_output__._sockets_dict
        assert set(outputs.keys()) == {"prompt"}
        assert outputs["prompt"].type == List[ChatMessage]

    def test_init_with_required_variables(self):
        builder = ChatPromptBuilder(
            template=[ChatMessage.from_user("This is a {{ variable }}")], required_variables=["variable"]
        )
        assert builder.required_variables == ["variable"]
        assert builder.template[0].content == "This is a {{ variable }}"
        assert builder._variables is None
        assert builder._required_variables == ["variable"]

        # we have inputs that contain: template, template_variables + inferred variables
        inputs = builder.__haystack_input__._sockets_dict
        assert set(inputs.keys()) == {"template", "template_variables", "variable"}
        assert inputs["template"].type == Optional[List[ChatMessage]]
        assert inputs["template_variables"].type == Optional[Dict[str, Any]]
        assert inputs["variable"].type == Any

        # response is always prompt
        outputs = builder.__haystack_output__._sockets_dict
        assert set(outputs.keys()) == {"prompt"}
        assert outputs["prompt"].type == List[ChatMessage]

    def test_init_with_custom_variables(self):
        variables = ["var1", "var2", "var3"]
        template = [ChatMessage.from_user("Hello, {{ var1 }}, {{ var2 }}!")]
        builder = ChatPromptBuilder(template=template, variables=variables)
        assert builder.required_variables == []
        assert builder._variables == variables
        assert builder.template[0].content == "Hello, {{ var1 }}, {{ var2 }}!"
        assert builder._required_variables is None

        # we have inputs that contain: template, template_variables + variables
        inputs = builder.__haystack_input__._sockets_dict
        assert set(inputs.keys()) == {"template", "template_variables", "var1", "var2", "var3"}
        assert inputs["template"].type == Optional[List[ChatMessage]]
        assert inputs["template_variables"].type == Optional[Dict[str, Any]]
        assert inputs["var1"].type == Any
        assert inputs["var2"].type == Any
        assert inputs["var3"].type == Any

        # response is always prompt
        outputs = builder.__haystack_output__._sockets_dict
        assert set(outputs.keys()) == {"prompt"}
        assert outputs["prompt"].type == List[ChatMessage]

    def test_run(self):
        builder = ChatPromptBuilder(template=[ChatMessage.from_user("This is a {{ variable }}")])
        res = builder.run(variable="test")
        assert res == {"prompt": [ChatMessage.from_user("This is a test")]}

    def test_run_template_variable(self):
        builder = ChatPromptBuilder(template=[ChatMessage.from_user("This is a {{ variable }}")])
        res = builder.run(template_variables={"variable": "test"})
        assert res == {"prompt": [ChatMessage.from_user("This is a test")]}

    def test_run_template_variable_overrides_variable(self):
        builder = ChatPromptBuilder(template=[ChatMessage.from_user("This is a {{ variable }}")])
        res = builder.run(template_variables={"variable": "test_from_template_var"}, variable="test")
        assert res == {"prompt": [ChatMessage.from_user("This is a test_from_template_var")]}

    def test_run_without_input(self):
        builder = ChatPromptBuilder(template=[ChatMessage.from_user("This is a template without input")])
        res = builder.run()
        assert res == {"prompt": [ChatMessage.from_user("This is a template without input")]}

    def test_run_with_missing_input(self):
        builder = ChatPromptBuilder(template=[ChatMessage.from_user("This is a {{ variable }}")])
        res = builder.run()
        assert res == {"prompt": [ChatMessage.from_user("This is a ")]}

    def test_run_with_missing_required_input(self):
        builder = ChatPromptBuilder(
            template=[ChatMessage.from_user("This is a {{ foo }}, not a {{ bar }}")], required_variables=["foo", "bar"]
        )
        with pytest.raises(ValueError, match="foo"):
            builder.run(bar="bar")
        with pytest.raises(ValueError, match="bar"):
            builder.run(foo="foo")
        with pytest.raises(ValueError, match="foo, bar"):
            builder.run()

    def test_run_with_variables(self):
        variables = ["var1", "var2", "var3"]
        template = [ChatMessage.from_user("Hello, {{ name }}! {{ var1 }}")]

        builder = ChatPromptBuilder(template=template, variables=variables)

        template_variables = {"name": "John"}
        expected_result = {"prompt": [ChatMessage.from_user("Hello, John! How are you?")]}

        assert builder.run(template_variables=template_variables, var1="How are you?") == expected_result

    def test_run_with_variables_and_runtime_template(self):
        variables = ["var1", "var2", "var3"]

        builder = ChatPromptBuilder(variables=variables)

        template = [ChatMessage.from_user("Hello, {{ name }}! {{ var1 }}")]
        template_variables = {"name": "John"}
        expected_result = {"prompt": [ChatMessage.from_user("Hello, John! How are you?")]}

        assert (
            builder.run(template=template, template_variables=template_variables, var1="How are you?")
            == expected_result
        )

    def test_run_overwriting_default_template(self):
        default_template = [ChatMessage.from_user("Hello, {{ name }}!")]

        builder = ChatPromptBuilder(template=default_template)

        template = [ChatMessage.from_user("Hello, {{ var1 }}{{ name }}!")]
        expected_result = {"prompt": [ChatMessage.from_user("Hello, John!")]}

        assert builder.run(template, name="John") == expected_result

    def test_run_overwriting_default_template_with_template_variables(self):
        default_template = [ChatMessage.from_user("Hello, {{ name }}!")]

        builder = ChatPromptBuilder(template=default_template)

        template = [ChatMessage.from_user("Hello, {{ var1 }} {{ name }}!")]
        template_variables = {"var1": "Big"}
        expected_result = {"prompt": [ChatMessage.from_user("Hello, Big John!")]}

        assert builder.run(template, template_variables, name="John") == expected_result

    def test_run_overwriting_default_template_with_variables(self):
        variables = ["var1", "var2", "name"]
        default_template = [ChatMessage.from_user("Hello, {{ name }}!")]

        builder = ChatPromptBuilder(template=default_template, variables=variables)

        template = [ChatMessage.from_user("Hello, {{ var1 }} {{ name }}!")]
        expected_result = {"prompt": [ChatMessage.from_user("Hello, Big John!")]}

        assert builder.run(template, name="John", var1="Big") == expected_result

    def test_run_with_invalid_template(self):
        builder = ChatPromptBuilder()

        template = [ChatMessage.from_user("Hello, {{ name }!")]
        template_variables = {"name": "John"}
        with pytest.raises(TemplateSyntaxError):
            builder.run(template, template_variables)

    def test_init_with_invalid_template(self):
        template = [ChatMessage.from_user("Hello, {{ name }!")]
        with pytest.raises(TemplateSyntaxError):
            ChatPromptBuilder(template)

    def test_run_without_template(self):
        prompt_builder = ChatPromptBuilder()
        with pytest.raises(
            ValueError, match="The ChatPromptBuilder requires a non-empty list of ChatMessage instances"
        ):
            prompt_builder.run()

    def test_run_with_empty_chat_message_list(self):
        prompt_builder = ChatPromptBuilder(template=[], variables=["documents"])
        with pytest.raises(
            ValueError, match="The ChatPromptBuilder requires a non-empty list of ChatMessage instances"
        ):
            prompt_builder.run()

    def test_chat_message_list_with_mixed_object_list(self):
        prompt_builder = ChatPromptBuilder(
            template=[ChatMessage.from_user("Hello"), "there world"], variables=["documents"]
        )
        with pytest.raises(
            ValueError, match="The ChatPromptBuilder expects a list containing only ChatMessage instances"
        ):
            prompt_builder.run()

    def test_provided_template_variables(self):
        prompt_builder = ChatPromptBuilder(variables=["documents"], required_variables=["city"])

        # both variables are provided
        prompt_builder._validate_variables({"name", "city"})

        # provided variables are a superset of the required variables
        prompt_builder._validate_variables({"name", "city", "age"})

        with pytest.raises(ValueError):
            prompt_builder._validate_variables({"name"})

    def test_example_in_pipeline(self):
        default_template = [
            ChatMessage.from_user("Here is the document: {{documents[0].content}} \\n Answer: {{query}}")
        ]
        prompt_builder = ChatPromptBuilder(template=default_template, variables=["documents"])

        @component
        class DocumentProducer:
            @component.output_types(documents=List[Document])
            def run(self, doc_input: str):
                return {"documents": [Document(content=doc_input)]}

        pipe = Pipeline()
        pipe.add_component("doc_producer", DocumentProducer())
        pipe.add_component("prompt_builder", prompt_builder)
        pipe.connect("doc_producer.documents", "prompt_builder.documents")

        template = [ChatMessage.from_user("Here is the document: {{documents[0].content}} \n Query: {{query}}")]
        result = pipe.run(
            data={
                "doc_producer": {"doc_input": "Hello world, I live in Berlin"},
                "prompt_builder": {
                    "template": template,
                    "template_variables": {"query": "Where does the speaker live?"},
                },
            }
        )

        assert result == {
            "prompt_builder": {
                "prompt": [
                    ChatMessage.from_user(
                        "Here is the document: Hello world, I live in Berlin \n Query: Where does the speaker live?"
                    )
                ]
            }
        }

    def test_example_in_pipeline_simple(self):
        default_template = [ChatMessage.from_user("This is the default prompt:\n Query: {{query}}")]
        prompt_builder = ChatPromptBuilder(template=default_template)

        pipe = Pipeline()
        pipe.add_component("prompt_builder", prompt_builder)

        # using the default prompt
        result = pipe.run(data={"query": "Where does the speaker live?"})
        expected_default = {
            "prompt_builder": {
                "prompt": [ChatMessage.from_user("This is the default prompt:\n Query: Where does the speaker live?")]
            }
        }
        assert result == expected_default

        # using the dynamic prompt
        result = pipe.run(
            data={
                "query": "Where does the speaker live?",
                "template": [ChatMessage.from_user("This is the dynamic prompt:\n Query: {{query}}")],
            }
        )
        expected_dynamic = {
            "prompt_builder": {
                "prompt": [ChatMessage.from_user("This is the dynamic prompt:\n Query: Where does the speaker live?")]
            }
        }
        assert result == expected_dynamic


class TestChatPromptBuilderDynamic:
    def test_multiple_templated_chat_messages(self):
        prompt_builder = ChatPromptBuilder()
        language = "French"
        location = "Berlin"
        messages = [
            ChatMessage.from_system("Write your response in this language:{{language}}"),
            ChatMessage.from_user("Tell me about {{location}}"),
        ]

        result = prompt_builder.run(template_variables={"language": language, "location": location}, template=messages)
        assert result["prompt"] == [
            ChatMessage.from_system("Write your response in this language:French"),
            ChatMessage.from_user("Tell me about Berlin"),
        ], "The templated messages should match the expected output."

    def test_multiple_templated_chat_messages_in_place(self):
        prompt_builder = ChatPromptBuilder()
        language = "French"
        location = "Berlin"
        messages = [
            ChatMessage.from_system("Write your response ins this language:{{language}}"),
            ChatMessage.from_user("Tell me about {{location}}"),
        ]

        res = prompt_builder.run(template_variables={"language": language, "location": location}, template=messages)
        assert res == {
            "prompt": [
                ChatMessage.from_system("Write your response ins this language:French"),
                ChatMessage.from_user("Tell me about Berlin"),
            ]
        }, "The templated messages should match the expected output."

    def test_some_templated_chat_messages(self):
        prompt_builder = ChatPromptBuilder()
        language = "English"
        location = "Paris"
        messages = [
            ChatMessage.from_system("Please, respond in the following language: {{language}}."),
            ChatMessage.from_user("I would like to learn more about {{location}}."),
            ChatMessage.from_assistant("Yes, I can help you with that {{subject}}"),
            ChatMessage.from_user("Ok so do so please, be elaborate."),
        ]

        result = prompt_builder.run(template_variables={"language": language, "location": location}, template=messages)

        expected_messages = [
            ChatMessage.from_system("Please, respond in the following language: English."),
            ChatMessage.from_user("I would like to learn more about Paris."),
            ChatMessage.from_assistant(
                "Yes, I can help you with that {{subject}}"
            ),  # assistant message should not be templated
            ChatMessage.from_user("Ok so do so please, be elaborate."),
        ]

        assert result["prompt"] == expected_messages, "The templated messages should match the expected output."

    def test_example_in_pipeline(self):
        prompt_builder = ChatPromptBuilder()

        pipe = Pipeline()
        pipe.add_component("prompt_builder", prompt_builder)

        location = "Berlin"
        system_message = ChatMessage.from_system(
            "You are a helpful assistant giving out valuable information to tourists."
        )
        messages = [system_message, ChatMessage.from_user("Tell me about {{location}}")]

        res = pipe.run(data={"prompt_builder": {"template_variables": {"location": location}, "template": messages}})
        assert res == {
            "prompt_builder": {
                "prompt": [
                    ChatMessage.from_system("You are a helpful assistant giving out valuable information to tourists."),
                    ChatMessage.from_user("Tell me about Berlin"),
                ]
            }
        }

        messages = [
            system_message,
            ChatMessage.from_user("What's the weather forecast for {{location}} in the next {{day_count}} days?"),
        ]

        res = pipe.run(
            data={
                "prompt_builder": {"template_variables": {"location": location, "day_count": "5"}, "template": messages}
            }
        )
        assert res == {
            "prompt_builder": {
                "prompt": [
                    ChatMessage.from_system("You are a helpful assistant giving out valuable information to tourists."),
                    ChatMessage.from_user("What's the weather forecast for Berlin in the next 5 days?"),
                ]
            }
        }

    def test_example_in_pipeline_with_multiple_templated_messages(self):
        # no parameter init, we don't use any runtime template variables
        prompt_builder = ChatPromptBuilder()

        pipe = Pipeline()
        pipe.add_component("prompt_builder", prompt_builder)

        location = "Berlin"
        system_message = ChatMessage.from_system(
            "You are a helpful assistant giving out valuable information to tourists in {{language}}."
        )
        messages = [system_message, ChatMessage.from_user("Tell me about {{location}}")]

        res = pipe.run(
            data={
                "prompt_builder": {
                    "template_variables": {"location": location, "language": "German"},
                    "template": messages,
                }
            }
        )
        assert res == {
            "prompt_builder": {
                "prompt": [
                    ChatMessage.from_system(
                        "You are a helpful assistant giving out valuable information to tourists in German."
                    ),
                    ChatMessage.from_user("Tell me about Berlin"),
                ]
            }
        }

        messages = [
            system_message,
            ChatMessage.from_user("What's the weather forecast for {{location}} in the next {{day_count}} days?"),
        ]

        res = pipe.run(
            data={
                "prompt_builder": {
                    "template_variables": {"location": location, "day_count": "5", "language": "English"},
                    "template": messages,
                }
            }
        )
        assert res == {
            "prompt_builder": {
                "prompt": [
                    ChatMessage.from_system(
                        "You are a helpful assistant giving out valuable information to tourists in English."
                    ),
                    ChatMessage.from_user("What's the weather forecast for Berlin in the next 5 days?"),
                ]
            }
        }

    def test_pipeline_complex(self):
        @component
        class ValueProducer:
            def __init__(self, value_to_produce: str):
                self.value_to_produce = value_to_produce

            @component.output_types(value_output=str)
            def run(self):
                return {"value_output": self.value_to_produce}

        pipe = Pipeline()
        pipe.add_component("prompt_builder", ChatPromptBuilder(variables=["value_output"]))
        pipe.add_component("value_producer", ValueProducer(value_to_produce="Berlin"))
        pipe.connect("value_producer.value_output", "prompt_builder")

        messages = [
            ChatMessage.from_system("You give valuable information to tourists."),
            ChatMessage.from_user("Tell me about {{value_output}}"),
        ]

        res = pipe.run(data={"template": messages})
        assert res == {
            "prompt_builder": {
                "prompt": [
                    ChatMessage.from_system("You give valuable information to tourists."),
                    ChatMessage.from_user("Tell me about Berlin"),
                ]
            }
        }

    def test_to_dict(self):
        component = ChatPromptBuilder(
            template=[ChatMessage.from_user("text and {var}"), ChatMessage.from_assistant("content {required_var}")],
            variables=["var", "required_var"],
            required_variables=["required_var"],
        )

        assert component.to_dict() == {
            "type": "haystack.components.builders.chat_prompt_builder.ChatPromptBuilder",
            "init_parameters": {
                "template": [
                    {"content": "text and {var}", "role": "user", "name": None, "meta": {}},
                    {"content": "content {required_var}", "role": "assistant", "name": None, "meta": {}},
                ],
                "variables": ["var", "required_var"],
                "required_variables": ["required_var"],
            },
        }

    def test_from_dict(self):
        component = ChatPromptBuilder.from_dict(
            data={
                "type": "haystack.components.builders.chat_prompt_builder.ChatPromptBuilder",
                "init_parameters": {
                    "template": [
                        {"content": "text and {var}", "role": "user", "name": None, "meta": {}},
                        {"content": "content {required_var}", "role": "assistant", "name": None, "meta": {}},
                    ],
                    "variables": ["var", "required_var"],
                    "required_variables": ["required_var"],
                },
            }
        )

        assert component.template == [
            ChatMessage.from_user("text and {var}"),
            ChatMessage.from_assistant("content {required_var}"),
        ]
        assert component._variables == ["var", "required_var"]
        assert component._required_variables == ["required_var"]

    def test_init_with_non_text_chat_message_content(self):
        messages = [
            ChatMessage.from_system("this is {{data}}"),
            ChatMessage.from_user("image_url:images.com/{{image}}"),
            ChatMessage.from_assistant(ByteStream(data=b"{{bytes}}", mime_type="image_base64/jpg")),
            ChatMessage.from_user(
                [
                    "this is {{text}}",
                    "image_url:more.images.com/{{more}}",
                    ByteStream(data=b"more {{content}}", mime_type="image_base64/jpg"),
                ]
            ),
        ]

        prompt_builder = ChatPromptBuilder(template=messages)

        assert prompt_builder.template == messages

    def test_run_with_non_text_chat_message_content(self):
        messages = [
            ChatMessage.from_system("this is {{data}}"),
            ChatMessage.from_user("image_url:images.com/{{image}}"),
            ChatMessage.from_assistant(ByteStream(data=b"{{bytes}}", mime_type="image_base64/png")),
            ChatMessage.from_user(
                [
                    "this is {{text}}",
                    "image_url:more.images.com/{{more}}",
                    ByteStream(data=b"more {{content}}", mime_type="image_base64/png"),
                ]
            ),
        ]

        prompt = ChatPromptBuilder().run(
            template=messages, data="TEXT", image="test.jpg", text="a string", more="apple.png", content="BYTES"
        )

        assert prompt["prompt"] == [
            ChatMessage.from_system("this is TEXT"),
            ChatMessage.from_user("image_url:images.com/test.jpg"),
            ChatMessage.from_assistant(ByteStream(data=b"{{bytes}}", mime_type="image_base64/png")),
            ChatMessage.from_user(
                [
                    "this is a string",
                    "image_url:more.images.com/apple.png",
                    ByteStream(data=b"more BYTES", mime_type="image_base64/png"),
                ]
            ),
        ]
