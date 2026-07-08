---
title: "Lara"
id: integrations-lara
description: "Lara integration for Haystack"
slug: "/integrations-lara"
---


## haystack_integrations.components.translators.lara.document_translator

### LaraDocumentTranslator

Translates the text content of Haystack Documents using translated's Lara translation API.

Lara is an adaptive translation AI that combines the fluency and context handling
of LLMs with low hallucination and latency. It adapts to domains at inference time
using optional context, instructions, translation memories, and glossaries. You can find
more detailed information in the [Lara documentation](https://developers.laratranslate.com/docs/introduction).

### Usage example

```python
from haystack import Document
from haystack.utils import Secret
from haystack_integrations.components.lara import LaraDocumentTranslator

translator = LaraDocumentTranslator(
    access_key_id=Secret.from_env_var("LARA_ACCESS_KEY_ID"),
    access_key_secret=Secret.from_env_var("LARA_ACCESS_KEY_SECRET"),
    source_lang="en-US",
    target_lang="de-DE",
)

doc = Document(content="Hello, world!")
result = translator.run(documents=[doc])
print(result["documents"][0].content)
```

#### __init__

```python
__init__(
    access_key_id: Secret = Secret.from_env_var("LARA_ACCESS_KEY_ID"),
    access_key_secret: Secret = Secret.from_env_var("LARA_ACCESS_KEY_SECRET"),
    source_lang: str | None = None,
    target_lang: str | None = None,
    context: str | None = None,
    instructions: str | None = None,
    style: Literal["faithful", "fluid", "creative"] = "faithful",
    adapt_to: list[str] | None = None,
    glossaries: list[str] | None = None,
    reasoning: bool = False,
)
```

Creats an instance of the LaraDocumentTranslator component.

**Parameters:**

- **access_key_id** (<code>Secret</code>) – Lara API access key ID. Defaults to the `LARA_ACCESS_KEY_ID` environment variable.
- **access_key_secret** (<code>Secret</code>) – Lara API access key secret. Defaults to the `LARA_ACCESS_KEY_SECRET` environment variable.
- **source_lang** (<code>str | None</code>) – Language code of the source text. If `None`, Lara auto-detects the source language.
  Use locale codes from the
  [supported languages list](https://developers.laratranslate.com/docs/supported-languages).
- **target_lang** (<code>str | None</code>) – Language code of the target text.
  Use locale codes from the
  [supported languages list](https://developers.laratranslate.com/docs/supported-languages).
- **context** (<code>str | None</code>) – Optional external context: text that is not translated but is sent to Lara to
  improve translation quality (e.g. surrounding sentences, prior messages).
  You can find more detailed information in the
  [Lara documentation](https://developers.laratranslate.com/docs/adapt-to-context).
- **instructions** (<code>str | None</code>) – Optional natural-language instructions to guide translation and
  specify domain-specific terminology (e.g. "Be formal", "Use a professional tone").
  You can find more detailed information in the
  [Lara documentation](https://developers.laratranslate.com/docs/adapt-to-instructions).
- **style** (<code>Literal['faithful', 'fluid', 'creative']</code>) – One of `"faithful"`, `"fluid"`, or `"creative"`.
  Default is `"faithful"`.
  Style description:
- `"faithful"`: For accuracy and precision. Keeps original structure and meaning.
  Ideal for manuals, legal documents.
- `"fluid"`: For readability and natural flow. Smooth, conversational. Good for general content.
- `"creative"`: For artistic and creative expression. Best for literature, marketing, or content
  where impact and tone matter more than literal wording.
  You can find more detailed information in the
  [Lara documentation](https://support.laratranslate.com/en/translation-styles).
- **adapt_to** (<code>list\[str\] | None</code>) – Optional list of translation memory IDs. Lara adapts to the style and terminology of these memories
  at inference time. Domain adaptation is available depending on your plan. You can find more
  detailed information in the
  [Lara documentation](https://developers.laratranslate.com/docs/adapt-to-translation-memories).
- **glossaries** (<code>list\[str\] | None</code>) – Optional list of glossary IDs. Lara applies these glossaries at inference time to enforce
  consistent terminology (e.g. brand names, product terms, legal or technical phrases) across translations.
  Glossary management and availability depends on your plan.
  You can find more detailed information in the
  [Lara documentation](https://developers.laratranslate.com/docs/manage-glossaries).
- **reasoning** (<code>bool</code>) – If `True`, uses the Lara Think model for higher-quality translation (multi-step linguistic analysis).
  Increases latency and cost. Availability depends on your plan. You can find more detailed information in the
  [Lara documentation](https://developers.laratranslate.com/docs/translate-text#reasoning-lara-think).

#### warm_up

```python
warm_up() -> None
```

Warm up the Lara translator by initializing the client.

#### run

```python
run(
    documents: list[Document],
    source_lang: str | list[str | None] | None = None,
    target_lang: str | list[str] | None = None,
    context: str | list[str] | None = None,
    instructions: str | list[str] | None = None,
    style: str | list[str] | None = None,
    adapt_to: list[str] | list[list[str]] | None = None,
    glossaries: list[str] | list[list[str]] | None = None,
    reasoning: bool | list[bool] | None = None,
) -> dict[str, list[Document]]
```

Translate the text content of each input Document using the Lara API.

Any of the translation parameters (source_lang, target_lang, context,
instructions, style, adapt_to, glossaries, reasoning) can be passed here
to override the defaults set when creating the component. They can be a single value
(applied to all documents) or a list of values with the same length as
`documents` for per-document settings.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – List of Haystack Documents whose `content` is to be translated.
- **source_lang** (<code>str | list\[str | None\] | None</code>) – Source language code(s). Use locale codes from the
  [supported languages list](https://developers.laratranslate.com/docs/supported-languages).
  If `None`, Lara auto-detects the source language. Single value or list (one per document).
- **target_lang** (<code>str | list\[str\] | None</code>) – Target language code(s). Use locale codes from the
  [supported languages list](https://developers.laratranslate.com/docs/supported-languages).
  Single value or list (one per document).
- **context** (<code>str | list\[str\] | None</code>) – Optional external context: text that is not translated but is sent to Lara to
  improve translation quality (e.g. surrounding sentences, prior messages).
  You can find more detailed information in the
  [Lara documentation](https://developers.laratranslate.com/docs/adapt-to-context).
- **instructions** (<code>str | list\[str\] | None</code>) – Optional natural-language instructions to guide translation and specify
  domain-specific terminology (e.g. "Be formal", "Use a professional tone").
  You can find more detailed information in the
  [Lara documentation](https://developers.laratranslate.com/docs/adapt-to-instructions).
- **style** (<code>str | list\[str\] | None</code>) – One of `"faithful"`, `"fluid"`, or `"creative"`.
  Style description:
- `"faithful"`: For accuracy and precision. Keeps original structure and meaning.
  Ideal for manuals, legal documents.
- `"fluid"`: For readability and natural flow. Smooth, conversational. Good for general content.
- `"creative"`: For artistic and creative expression. Best for literature, marketing, or content
  where impact and tone matter more than literal wording.
  You can find more detailed information in the
  [Lara documentation](https://support.laratranslate.com/en/translation-styles).
- **adapt_to** (<code>list\[str\] | list\[list\[str\]\] | None</code>) – Optional list of translation memory IDs. Lara adapts to the style and terminology
  of these memories at inference time. Domain adaptation is available depending on your plan.
  You can find more detailed information in the
  [Lara documentation](https://developers.laratranslate.com/docs/adapt-to-translation-memories).
- **glossaries** (<code>list\[str\] | list\[list\[str\]\] | None</code>) – Optional list of glossary IDs. Lara applies these glossaries at inference time to enforce
  consistent terminology (e.g. brand names, product terms, legal or technical phrases) across translations.
  Glossary management and availability depends on your plan.
  You can find more detailed information in the
  [Lara documentation](https://developers.laratranslate.com/docs/manage-glossaries).
- **reasoning** (<code>bool | list\[bool\] | None</code>) – If `True`, uses the Lara Think model for higher-quality translation (multi-step linguistic analysis).
  Increases latency and cost. Availability depends on your plan. You can find more detailed information in the
  [Lara documentation](https://developers.laratranslate.com/docs/translate-text#reasoning-lara-think).

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following keys:
- `documents`: A list of translated documents.

**Raises:**

- <code>ValueError</code> – If any list-valued parameter has length != `len(documents)`.
