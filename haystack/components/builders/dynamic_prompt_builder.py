import logging
from typing import Dict, Any, Optional, List, Union, Set

from jinja2 import Template, meta

from haystack import component
from haystack import default_to_dict
from haystack.dataclasses.chat_message import ChatMessage, ChatRole

logger = logging.getLogger(__name__)


@component
class DynamicPromptBuilder:
    """
    DynamicPromptBuilder is designed to construct dynamic prompts by processing either a list of `ChatMessage`
    instances or a string template. It integrates with Jinja2 templating for dynamic prompt generation.

    In the case of `ChatMessage` instances, DynamicPromptBuilder assumes the last user message in the list as a
    template and renders it with resolved pipeline variables and any additional template variables provided. For a
    string template, it applies the template variables directly to render the final prompt. This dual functionality
    allows DynamicPromptBuilder to be versatile in handling different types of prompt sources, making it suitable for
    both chat-based and non-chat-based prompt generation scenarios.

    You can provide additional template variables directly to the pipeline `run` method. They are then merged with the
    variables resolved from the pipeline runtime. This allows for greater flexibility and customization of the
    generated prompts based on runtime conditions and user inputs.

    The following example demonstrates how to use DynamicPromptBuilder to generate a chat prompt:

    ```python
    from haystack.components.builders import DynamicPromptBuilder
    from haystack.components.generators.chat import GPTChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack import Pipeline

    # no parameter init, we don't use any runtime template variables
    prompt_builder = DynamicPromptBuilder()
    llm = GPTChatGenerator(api_key="<your-api-key>", model_name="gpt-3.5-turbo")

    pipe = Pipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.connect("prompt_builder.prompt", "llm.messages")

    location = "Berlin"
    system_message = ChatMessage.from_system("You are a helpful assistant giving out valuable information to tourists.")
    messages = [system_message, ChatMessage.from_user("Tell me about {{location}}")]


    res = pipe.run(data={"prompt_builder": {"template_variables": {"location": location}, "prompt_source": messages}})
    print(res)

    >> {'llm': {'replies': [ChatMessage(content="Berlin is the capital city of Germany and one of the most vibrant
    and diverse cities in Europe. Here are some key things to know...Enjoy your time exploring the vibrant and dynamic
    capital of Germany!", role=<ChatRole.ASSISTANT: 'assistant'>, name=None, meta={'model': 'gpt-3.5-turbo-0613',
    'index': 0, 'finish_reason': 'stop', 'usage': {'prompt_tokens': 27, 'completion_tokens': 681, 'total_tokens': 708}})]}}


    messages = [system_message, ChatMessage.from_user("What's the weather forecast for {{location}} in the next {{day_count}} days?")]

    res = pipe.run(data={"prompt_builder": {"template_variables": {"location": location, "day_count": "5"},
                                        "prompt_source": messages}})

    print(res)
    >> {'llm': {'replies': [ChatMessage(content="Here is the weather forecast for Berlin in the next 5
    days:\n\nDay 1: Mostly cloudy with a high of 22°C (72°F) and...so it's always a good idea to check for updates
    closer to your visit.", role=<ChatRole.ASSISTANT: 'assistant'>, name=None, meta={'model': 'gpt-3.5-turbo-0613',
    'index': 0, 'finish_reason': 'stop', 'usage': {'prompt_tokens': 37, 'completion_tokens': 201, 'total_tokens': 238}})]}}

    ```

    The primary advantage of using DynamicPromptBuilder is showcased in the above provided examples. DynamicPromptBuilder
    allows dynamic customization of prompt messages without the need to reload/recreate the pipeline for each invocation.

    In the example, the first query asks for general information about Berlin, and the second query requests the weather
    forecast for Berlin in the next few days. DynamicPromptBuilder efficiently handles these distinct prompt structures
    through adjustments in the pipeline run parameters invocations. This is in stark contrast to a regular PromptBuilder,
    which would require recreating/reloading the pipeline for each distinct type of query, leading to inefficiency and
    potential service disruptions, especially in server environments where continuous service is vital.

    Note of course that weather forecast in the example above is fictional, but it can be easily connected to a weather
    API to provide real weather forecasts.


    The following example demonstrates how to use DynamicPromptBuilder to generate a chat prompt with resolution
    of pipeline runtime variables (such as documents):

    ```python
    from haystack.components.builders import DynamicPromptBuilder
    from haystack.components.generators.chat import GPTChatGenerator
    from haystack.dataclasses import ChatMessage, Document
    from haystack import Pipeline, component
    from typing import List

    # we'll use documents runtime variable in our template, so we need to specify it in the init
    prompt_builder = DynamicPromptBuilder(runtime_variables=["documents"])
    llm = GPTChatGenerator(api_key="<your-api-key>", model_name="gpt-3.5-turbo")


    @component
    class DocumentProducer:

        @component.output_types(documents=List[Document])
        def run(self, doc_input: str):
            return {"documents": [Document(content=doc_input)]}



    pipe = Pipeline()
    pipe.add_component("doc_producer", DocumentProducer())
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)

    # note here how prompt_builder.documents is received from doc_producer.documents
    pipe.connect("doc_producer.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "llm.messages")

    messages = [ChatMessage.from_system("Be helpful assistant, but brief!"),
                ChatMessage.from_user("Here is the document: {{documents[0].content}} Now, answer the
                following: {{query}}")]


    pipe.run(data={"doc_producer": {"doc_input": "Hello world, I'm Haystack!"},
                   "prompt_builder": {"prompt_source": messages,
                                      "template_variables":{"query": "who's making a greeting?"}}})

    >> {'llm': {'replies': [ChatMessage(content='Haystack', role=<ChatRole.ASSISTANT: 'assistant'>, name=None,
    >> meta={'model': 'gpt-3.5-turbo-0613', 'index': 0, 'finish_reason': 'stop', 'usage':
    >> {'prompt_tokens': 51, 'completion_tokens': 2, 'total_tokens': 53}})]}}
    ```

    Similarly to chat prompt generation, you can use DynamicPromptBuilder to generate non-chat-based prompts.
    The following example demonstrates how to use DynamicPromptBuilder to generate a non-chat prompt:

    ```python
    prompt_builder = DynamicPromptBuilder(runtime_variables=["documents"], chat_mode=False)
    llm = GPTGenerator(api_key="<your-api-key>", model_name="gpt-3.5-turbo")


    @component
    class DocumentProducer:

      @component.output_types(documents=List[Document])
      def run(self, doc_input: str):
        return {"documents": [Document(content=doc_input)]}


    pipe = Pipeline()
    pipe.add_component("doc_producer", DocumentProducer())
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.connect("doc_producer.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "llm.prompt")

    template = "Here is the document: {{documents[0].content}} \n Answer: {{query}}"
    pipe.run(data={"doc_producer": {"doc_input": "Hello world, I live in Berlin"},
               "prompt_builder": {"prompt_source": template,
                                  "template_variables":{"query": "Where does the speaker live?"}}})

    >> {'llm': {'replies': ['The speaker lives in Berlin.'],
    >> 'meta': [{'model': 'gpt-3.5-turbo-0613',
    >> 'index': 0,
    >> 'finish_reason': 'stop',
    >> 'usage': {'prompt_tokens': 28,
    >> 'completion_tokens': 6,
    >> 'total_tokens': 34}}]}}

    """

    def __init__(self, runtime_variables: Optional[List[str]] = None, chat_mode: Optional[bool] = True):
        """
        Initializes DynamicPromptBuilder with the provided variable names. These variable names are used to resolve
        variables and their values during pipeline runtime execution. For example, if `runtime_variables` contains
        `documents` your instance of DynamicPromptBuilder will expect an input called `documents`.
        The values associated with variables from the pipeline runtime are then injected into template placeholders
        of either a ChatMessage or a string template that is provided to the `run` method.
        See `run` method for more details.

        :param runtime_variables: A list of template variable names you can use in chat prompt construction.
        :type runtime_variables: Optional[List[str]]
        :param chat_mode: A boolean flag to indicate if the chat prompt is being built for a chat-based prompt
        templating. Defaults to True.
        :type chat_mode: Optional[bool]
        """
        runtime_variables = runtime_variables or []

        # setup inputs
        if chat_mode:
            run_input_slots = {"prompt_source": List[ChatMessage], "template_variables": Optional[Dict[str, Any]]}
        else:
            run_input_slots = {"prompt_source": str, "template_variables": Optional[Dict[str, Any]]}

        kwargs_input_slots = {var: Optional[Any] for var in runtime_variables}
        component.set_input_types(self, **run_input_slots, **kwargs_input_slots)

        # setup outputs
        if chat_mode:
            component.set_output_types(self, prompt=List[ChatMessage])
        else:
            component.set_output_types(self, prompt=str)

        self.runtime_variables = runtime_variables
        self.chat_mode = chat_mode

    def to_dict(self) -> Dict[str, Any]:
        """
         Converts the `DynamicPromptBuilder` instance to a dictionary format, primarily for serialization purposes.

        :return: A dictionary representation of the `DynamicPromptBuilder` instance, including its template variables.
        :rtype: Dict[str, Any]
        """
        return default_to_dict(self, runtime_variables=self.runtime_variables, chat_mode=self.chat_mode)

    def run(
        self,
        prompt_source: Union[List[ChatMessage], str],
        template_variables: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Executes the dynamic prompt building process. Depending on the provided type of `prompt_source`, this method
        either processes a list of `ChatMessage` instances or a string template. In the case of `ChatMessage` instances,
        the last user message is treated as a template and rendered with the resolved pipeline variables and any
        additional template variables provided. For a string template, it directly applies the template variables to
        render the final prompt. You can provide additional template variables directly to this method, that are then merged
        with the variables resolved from the pipeline runtime.

        :param prompt_source: A list of `ChatMessage` instances or a string template. The list scenario assumes the last
        user message as the template for the chat prompt, while the string scenario is used for non-chat-based prompts.
        :type prompt_source: Union[List[ChatMessage], str]

        :param template_variables: An optional dictionary of template variables. Template variables provided at
        initialization are required to resolve pipeline variables, and these are additional variables users can
        provide directly to this method.
        :type template_variables: Optional[Dict[str, Any]]

        :param kwargs: Additional keyword arguments, typically resolved from a pipeline, which are merged with the
        provided template variables.

        :return: A dictionary containing the key "prompt", which holds either the updated list of `ChatMessage`
        instances or the rendered string template, forming the complete dynamic prompt.
        :rtype: Dict[str, Union[List[ChatMessage], str]]
        """
        kwargs = kwargs or {}
        template_variables = template_variables or {}
        template_variables_combined = {**kwargs, **template_variables}
        if not template_variables_combined:
            raise ValueError(
                "The DynamicPromptBuilder run method requires template variables, but none were provided. "
                "Please provide an appropriate template variable to enable prompt generation."
            )
        # some of these checks are superfluous because pipeline will check them as well but let's
        # handle them anyway for better error messages and robustness
        result: Union[List[ChatMessage], str]
        if isinstance(prompt_source, str):
            result = self._process_simple_template(prompt_source, template_variables_combined)
        elif isinstance(prompt_source, list):
            result = self._process_chat_messages(prompt_source, template_variables_combined)
        else:
            raise ValueError(
                f"{self.__class__.__name__} was not provided with a list of ChatMessage(s) or a string template."
                "Please check the parameters passed to its run method."
            )
        return {"prompt": result}

    def _process_simple_template(self, prompt_source: str, template_variables: Dict[str, Any]) -> str:
        """
        Renders the template from the provided string source with the provided template variables.

        :param prompt_source: A Jinja2 template as a string.
        :type prompt_source: str
        :param template_variables: A dictionary of template variables.
        :type template_variables: Dict[str, Any]
        :return: A string containing the rendered template.
        :rtype: str
        """
        template = self._validate_template(prompt_source, set(template_variables.keys()))
        return template.render(template_variables)

    def _process_chat_messages(self, prompt_source: List[ChatMessage], template_variables: Dict[str, Any]):
        """
        Processes a list of :class:`ChatMessage` instances to generate a chat prompt.

        It takes the last user message in the list, treats it as a template, and renders it with the provided
        template variables. The resulting message replaces the last user message in the list, forming a complete,
        templated chat prompt.

        :param prompt_source: A list of `ChatMessage` instances to be processed. The last message is expected
        to be from a user and is treated as a template.
        :type prompt_source: List[ChatMessage]

        :param template_variables: A dictionary of template variables used for rendering the last user message.
        :type template_variables: Dict[str, Any]

        :return: A list of `ChatMessage` instances, where the last user message has been replaced with its
        templated version.
        :rtype: List[ChatMessage]

        :raises ValueError: If `chat_messages` is empty or contains elements that are not instances of
        `ChatMessage`.
        :raises ValueError: If the last message in `chat_messages` is not from a user.
        """
        if not prompt_source:
            raise ValueError(
                f"The {self.__class__.__name__} requires a non-empty list of ChatMessage instances. "
                f"Please provide a valid list of ChatMessage instances to render the prompt."
            )
        if not all(isinstance(message, ChatMessage) for message in prompt_source):
            raise ValueError(
                f"The {self.__class__.__name__} expects a list containing only ChatMessage instances. "
                f"The provided list contains other types. Please ensure that all elements in the list "
                f"are ChatMessage instances."
            )

        last_message: ChatMessage = prompt_source[-1]
        if last_message.is_from(ChatRole.USER):
            template = self._validate_template(last_message.content, set(template_variables.keys()))
            templated_user_message = ChatMessage.from_user(template.render(template_variables))
            return prompt_source[:-1] + [templated_user_message]
        else:
            logger.warning(
                "DynamicPromptBuilder was not provided with a user message as the last message in "
                "chat conversation, no templating will be applied."
            )
            return prompt_source

    def _validate_template(self, template_text: str, provided_variables: Set[str]):
        """
        Checks if all the required template variables are provided to the pipeline `run` method.
        If all the required template variables are provided, returns a Jinja2 template object.
        Otherwise, raises a ValueError.

        :param template_text: A Jinja2 template as a string.
        :param provided_variables: A set of provided template variables.
        :type provided_variables: Set[str]
        :return: A Jinja2 template object if all the required template variables are provided.
        :raises ValueError: If all the required template variables are not provided.
        """
        template = Template(template_text)
        ast = template.environment.parse(template_text)
        required_template_variables = meta.find_undeclared_variables(ast)
        filled_template_vars = required_template_variables.intersection(provided_variables)
        if len(filled_template_vars) != len(required_template_variables):
            raise ValueError(
                f"The {self.__class__.__name__} requires specific template variables that are missing. "
                f"Required variables: {required_template_variables}. Only the following variables were "
                f"provided: {provided_variables}. Please provide all the required template variables."
            )
        return template
