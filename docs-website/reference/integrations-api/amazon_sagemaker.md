---
title: "Amazon Sagemaker"
id: integrations-amazon-sagemaker
description: "Amazon Sagemaker integration for Haystack"
slug: "/integrations-amazon-sagemaker"
---


## `haystack_integrations.components.generators.amazon_sagemaker.sagemaker`

### `SagemakerGenerator`

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

#### `__init__`

```python
__init__(
    model: str,
    aws_access_key_id: Secret | None = Secret.from_env_var(
        ["AWS_ACCESS_KEY_ID"], strict=False
    ),
    aws_secret_access_key: Secret | None = Secret.from_env_var(
        ["AWS_SECRET_ACCESS_KEY"], strict=False
    ),
    aws_session_token: Secret | None = Secret.from_env_var(
        ["AWS_SESSION_TOKEN"], strict=False
    ),
    aws_region_name: Secret | None = Secret.from_env_var(
        ["AWS_DEFAULT_REGION"], strict=False
    ),
    aws_profile_name: Secret | None = Secret.from_env_var(
        ["AWS_PROFILE"], strict=False
    ),
    aws_custom_attributes: dict[str, Any] | None = None,
    generation_kwargs: dict[str, Any] | None = None,
)
```

Instantiates the session with SageMaker.

**Parameters:**

- **aws_access_key_id** (<code>Secret | None</code>) – The `Secret` for AWS access key ID.
- **aws_secret_access_key** (<code>Secret | None</code>) – The `Secret` for AWS secret access key.
- **aws_session_token** (<code>Secret | None</code>) – The `Secret` for AWS session token.
- **aws_region_name** (<code>Secret | None</code>) – The `Secret` for AWS region name. If not provided, the default region will be used.
- **aws_profile_name** (<code>Secret | None</code>) – The `Secret` for AWS profile name. If not provided, the default profile will be used.
- **model** (<code>str</code>) – The name for SageMaker Model Endpoint.
- **aws_custom_attributes** (<code>dict\[str, Any\] | None</code>) – Custom attributes to be passed to SageMaker, for example `{"accept_eula": True}`
  in case of Llama-2 models.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for text generation. For a list of supported parameters
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

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> SagemakerGenerator
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>SagemakerGenerator</code> – Deserialized component.

#### `run`

```python
run(
    prompt: str, generation_kwargs: dict[str, Any] | None = None
) -> dict[str, list[str] | list[dict[str, Any]]]
```

Invoke the text generation inference based on the provided prompt and generation parameters.

**Parameters:**

- **prompt** (<code>str</code>) – The string prompt to use for text generation.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for text generation. These parameters will
  potentially override the parameters passed in the `__init__` method.

**Returns:**

- <code>dict\[str, list\[str\] | list\[dict\[str, Any\]\]\]</code> – A dictionary with the following keys:
- `replies`: A list of strings containing the generated responses
- `meta`: A list of dictionaries containing the metadata for each response.

**Raises:**

- <code>ValueError</code> – If the model response type is not a list of dictionaries or a single dictionary.
- <code>SagemakerNotReadyError</code> – If the SageMaker model is not ready to accept requests.
- <code>SagemakerInferenceError</code> – If the SageMaker Inference returns an error.
