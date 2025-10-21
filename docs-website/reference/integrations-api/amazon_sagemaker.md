---
title: "Amazon Sagemaker"
id: integrations-amazon-sagemaker
description: "Amazon Sagemaker integration for Haystack"
slug: "/integrations-amazon-sagemaker"
---

<a id="haystack_integrations.components.generators.amazon_sagemaker.sagemaker"></a>

# Module haystack\_integrations.components.generators.amazon\_sagemaker.sagemaker

<a id="haystack_integrations.components.generators.amazon_sagemaker.sagemaker.SagemakerGenerator"></a>

## SagemakerGenerator

Enables text generation using Amazon Sagemaker.

SagemakerGenerator supports Large Language Models (LLMs) hosted and deployed on a SageMaker Inference Endpoint.
For guidance on how to deploy a model to SageMaker, refer to the
[SageMaker JumpStart foundation models documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-use.html).

Usage example:
```python
# Make sure your AWS credentials are set up correctly. You can use environment variables or a shared credentials
# file. Then you can use the generator as follows:
from haystack_integrations.components.generators.amazon_sagemaker import SagemakerGenerator

generator = SagemakerGenerator(model="jumpstart-dft-hf-llm-falcon-7b-bf16")
response = generator.run("What's Natural Language Processing? Be brief.")
print(response)
>>> {'replies': ['Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on
>>> the interaction between computers and human language. It involves enabling computers to understand, interpret,
>>> and respond to natural human language in a way that is both meaningful and useful.'], 'meta': [{}]}
```

<a id="haystack_integrations.components.generators.amazon_sagemaker.sagemaker.SagemakerGenerator.__init__"></a>

#### SagemakerGenerator.\_\_init\_\_

```python
def __init__(
        model: str,
        aws_access_key_id: Optional[Secret] = Secret.from_env_var(
            ["AWS_ACCESS_KEY_ID"], strict=False),
        aws_secret_access_key: Optional[Secret] = Secret.
    from_env_var(  # noqa: B008
        ["AWS_SECRET_ACCESS_KEY"], strict=False),
        aws_session_token: Optional[Secret] = Secret.from_env_var(
            ["AWS_SESSION_TOKEN"], strict=False),
        aws_region_name: Optional[Secret] = Secret.from_env_var(
            ["AWS_DEFAULT_REGION"], strict=False),
        aws_profile_name: Optional[Secret] = Secret.from_env_var(
            ["AWS_PROFILE"], strict=False),
        aws_custom_attributes: Optional[Dict[str, Any]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None)
```

Instantiates the session with SageMaker.

**Arguments**:

- `aws_access_key_id`: The `Secret` for AWS access key ID.
- `aws_secret_access_key`: The `Secret` for AWS secret access key.
- `aws_session_token`: The `Secret` for AWS session token.
- `aws_region_name`: The `Secret` for AWS region name. If not provided, the default region will be used.
- `aws_profile_name`: The `Secret` for AWS profile name. If not provided, the default profile will be used.
- `model`: The name for SageMaker Model Endpoint.
- `aws_custom_attributes`: Custom attributes to be passed to SageMaker, for example `{"accept_eula": True}`
in case of Llama-2 models.
- `generation_kwargs`: Additional keyword arguments for text generation. For a list of supported parameters
see your model's documentation page, for example here for HuggingFace models:
https://huggingface.co/blog/sagemaker-huggingface-llm#4-run-inference-and-chat-with-our-model

Specifically, Llama-2 models support the following inference payload parameters:

- `max_new_tokens`: Model generates text until the output length (excluding the input context length)
    reaches `max_new_tokens`. If specified, it must be a positive integer.
- `temperature`: Controls the randomness in the output. Higher temperature results in output sequence with
    low-probability words and lower temperature results in output sequence with high-probability words.
    If `temperature=0`, it results in greedy decoding. If specified, it must be a positive float.
- `top_p`: In each step of text generation, sample from the smallest possible set of words with cumulative
    probability `top_p`. If specified, it must be a float between 0 and 1.
- `return_full_text`: If `True`, input text will be part of the output generated text. If specified, it must
    be boolean. The default value for it is `False`.

<a id="haystack_integrations.components.generators.amazon_sagemaker.sagemaker.SagemakerGenerator.to_dict"></a>

#### SagemakerGenerator.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.generators.amazon_sagemaker.sagemaker.SagemakerGenerator.from_dict"></a>

#### SagemakerGenerator.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "SagemakerGenerator"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.generators.amazon_sagemaker.sagemaker.SagemakerGenerator.run"></a>

#### SagemakerGenerator.run

```python
@component.output_types(replies=List[str], meta=List[Dict[str, Any]])
def run(
    prompt: str,
    generation_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Union[List[str], List[Dict[str, Any]]]]
```

Invoke the text generation inference based on the provided prompt and generation parameters.

**Arguments**:

- `prompt`: The string prompt to use for text generation.
- `generation_kwargs`: Additional keyword arguments for text generation. These parameters will
potentially override the parameters passed in the `__init__` method.

**Raises**:

- `ValueError`: If the model response type is not a list of dictionaries or a single dictionary.
- `SagemakerNotReadyError`: If the SageMaker model is not ready to accept requests.
- `SagemakerInferenceError`: If the SageMaker Inference returns an error.

**Returns**:

A dictionary with the following keys:
- `replies`: A list of strings containing the generated responses
- `meta`: A list of dictionaries containing the metadata for each response.
