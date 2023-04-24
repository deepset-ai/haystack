from typing import Dict, Optional

from haystack.nodes.prompt.invocation_layer.open_ai import OpenAIInvocationLayer


class AzureOpenAIInvocationLayer(OpenAIInvocationLayer):
    """
    Azure OpenAI Invocation Layer

    This layer is used to invoke the OpenAI API on Azure. It is essentially the same as the OpenAIInvocationLayer
    with additional two parameters: `azure_base_url` and `azure_deployment_name`. The `azure_base_url` is the URL of the Azure OpenAI
    endpoint and the `azure_deployment_name` is the name of the deployment.
    """

    def __init__(
        self,
        azure_base_url: str,
        azure_deployment_name: str,
        api_key: str,
        api_version: str = "2022-12-01",
        model_name_or_path: str = "text-davinci-003",
        max_length: Optional[int] = 100,
        **kwargs,
    ):
        super().__init__(api_key, model_name_or_path, max_length, **kwargs)
        self.azure_base_url = azure_base_url
        self.azure_deployment_name = azure_deployment_name
        self.api_version = api_version

    @property
    def url(self) -> str:
        return f"{self.azure_base_url}/openai/deployments/{self.azure_deployment_name}/completions?api-version={self.api_version}"

    @property
    def headers(self) -> Dict[str, str]:
        return {"api-key": self.api_key, "Content-Type": "application/json"}

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        """
        Ensures Azure OpenAI Invocation Layer is selected when `azure_base_url` and `azure_deployment_name` are provided in
        addition to a list of supported models.
        """
        valid_model = any(m for m in ["ada", "babbage", "davinci", "curie"] if m in model_name_or_path)
        return (
            valid_model and kwargs.get("azure_base_url") is not None and kwargs.get("azure_deployment_name") is not None
        )
