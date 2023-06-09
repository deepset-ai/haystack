from abc import abstractmethod
from typing import Dict, List, Union, Type


class PromptModelInvocationLayer:
    """
    PromptModelInvocationLayer implementations execute a prompt on an underlying model.

    The implementation can be a simple invocation on the underlying model running in a local runtime, or
    could be even remote, for example, a call to a remote API endpoint.
    """

    invocation_layer_providers: List[Type["PromptModelInvocationLayer"]] = []

    def __init__(self, model_name_or_path: str, **kwargs):
        """
        Creates a new PromptModelInvocationLayer instance.

        :param model_name_or_path: The name or path of the underlying model.
        :param kwargs: Additional keyword arguments passed to the underlying model.
        """
        if model_name_or_path is None or len(model_name_or_path) == 0:
            raise ValueError("model_name_or_path cannot be None or empty string")

        self.model_name_or_path = model_name_or_path

    def __init_subclass__(cls, **kwargs):
        """
        Used to register user-defined invocation layers.

        Called when a subclass of PromptModelInvocationLayer is imported.
        """
        super().__init_subclass__(**kwargs)
        cls.invocation_layer_providers.append(cls)

    @abstractmethod
    def invoke(self, *args, **kwargs):
        """
        It takes a prompt and returns a list of generated text using the underlying model.
        :return: A list of generated text.
        """
        pass

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        """
        Checks if the given model is supported by this invocation layer.

        :param model_name_or_path: The name or path of the model.
        :param kwargs: Additional keyword arguments passed to the underlying model which might be used to determine
        if the model is supported.
        :return: True if this invocation layer supports the model, False otherwise.
        """
        return False

    @abstractmethod
    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        """Ensure that length of the prompt and answer is within the maximum token length of the PromptModel.

        :param prompt: Prompt text to be sent to the generative model.
        """
        pass
