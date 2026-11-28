---
title: "Presidio"
id: integrations-presidio
description: "Presidio integration for Haystack"
slug: "/integrations-presidio"
---


## haystack_integrations.components.extractors.presidio.presidio_entity_extractor

### PresidioEntityExtractor

Detects PII entities in Haystack Documents using Microsoft Presidio Analyzer.

See [Presidio Analyzer](https://microsoft.github.io/presidio/) for details.

Accepts a list of Documents and returns new Documents with detected PII entities stored
in each Document's metadata under the key `"entities"`. Each entry in the list contains
the entity type, start/end character offsets, and the confidence score.

Original Documents are not mutated. Documents without text content are passed through unchanged.

The analyzer engine is loaded on the first call to `run()`,
or by calling `warm_up()` explicitly beforehand.

### Usage example

```python
from haystack import Document
from haystack_integrations.components.extractors.presidio import PresidioEntityExtractor

extractor = PresidioEntityExtractor()
result = extractor.run(documents=[Document(content="Contact Alice at alice@example.com")])
print(result["documents"][0].meta["entities"])
# [{"entity_type": "PERSON", "start": 8, "end": 13, "score": 0.85},
#  {"entity_type": "EMAIL_ADDRESS", "start": 17, "end": 34, "score": 1.0}]
```

#### SPACY_DEFAULT_MODELS

```python
SPACY_DEFAULT_MODELS: dict[str, str] = _SPACY_DEFAULT_MODELS
```

Mapping from ISO 639-1 language code to the largest available spaCy model for that language.

Used to automatically select an NLP model when `models` is not specified.
See [spaCy documentation](https://spacy.io/models) for the full list of available spaCy models.

#### __init__

```python
__init__(
    *,
    language: str = "en",
    entities: list[str] | None = None,
    score_threshold: float = 0.35,
    models: list[dict[str, str]] | None = None
) -> None
```

Initializes the PresidioEntityExtractor.

**Parameters:**

- **language** (<code>str</code>) – ISO 639-1 language code for PII detection. Defaults to `"en"`.
  For languages in the built-in mapping (e.g. `"de"`, `"fr"`, `"es"`), the appropriate
  spaCy model is loaded automatically at warm-up time — no need to set `models`.
  For unsupported languages, use the `models` parameter to configure a custom model.
  See [Presidio supported languages](https://microsoft.github.io/presidio/analyzer/languages/).
- **entities** (<code>list\[str\] | None</code>) – List of PII entity types to detect (e.g. `["PERSON", "EMAIL_ADDRESS"]`).
  If `None`, all supported entity types are detected.
  See [Presidio supported entities](https://microsoft.github.io/presidio/supported_entities/).
- **score_threshold** (<code>float</code>) – Minimum confidence score (0-1) for a detected entity to be included. Defaults to `0.35`.
  See [Presidio analyzer documentation](https://microsoft.github.io/presidio/analyzer/).
- **models** (<code>list\[dict\[str, str\]\] | None</code>) – Advanced override: list of spaCy model configurations.
  Each entry must contain `"lang_code"` and `"model_name"` keys,
  e.g. `[{"lang_code": "fr", "model_name": "fr_core_news_md"}]`.
  Use this only when you need a specific model variant or a language not covered by the
  built-in mapping. If `None`, the model is selected automatically from `SPACY_DEFAULT_MODELS`
  based on `language`.

#### warm_up

```python
warm_up() -> None
```

Initializes the Presidio analyzer engine.

This method loads the underlying NLP models. In a Haystack Pipeline,
this is called automatically before the first run.

#### run

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Detects PII entities in the provided Documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – List of Documents to analyze for PII entities.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with key `documents` containing Documents with detected entities
  stored in metadata under the key `"entities"`.

## haystack_integrations.components.preprocessors.presidio.presidio_document_cleaner

### PresidioDocumentCleaner

Anonymizes PII in Haystack Documents using [Microsoft Presidio](https://microsoft.github.io/presidio/).

Accepts a list of Documents, detects personally identifiable information (PII) in their
text content, and returns new Documents with PII replaced by entity type placeholders
(e.g. `<PERSON>`, `<EMAIL_ADDRESS>`). Original Documents are not mutated.

Documents without text content are passed through unchanged.

The analyzer and anonymizer engines are loaded on the first call to `run()`,
or by calling `warm_up()` explicitly beforehand.

### Usage example

```python
from haystack import Document
from haystack_integrations.components.preprocessors.presidio import PresidioDocumentCleaner

cleaner = PresidioDocumentCleaner()
result = cleaner.run(documents=[Document(content="My name is John and my email is john@example.com")])
print(result["documents"][0].content)
# My name is <PERSON> and my email is <EMAIL_ADDRESS>
```

#### SPACY_DEFAULT_MODELS

```python
SPACY_DEFAULT_MODELS: dict[str, str] = _SPACY_DEFAULT_MODELS
```

Mapping from ISO 639-1 language code to the largest available spaCy model for that language.

Used to automatically select an NLP model when `models` is not specified.
See [spaCy documentation](https://spacy.io/models) for the full list of available spaCy models.

#### __init__

```python
__init__(
    *,
    language: str = "en",
    entities: list[str] | None = None,
    score_threshold: float = 0.35,
    models: list[dict[str, str]] | None = None
) -> None
```

Initializes the PresidioDocumentCleaner.

**Parameters:**

- **language** (<code>str</code>) – ISO 639-1 language code for PII detection. Defaults to `"en"`.
  For languages in the built-in mapping (e.g. `"de"`, `"fr"`, `"es"`), the appropriate
  spaCy model is loaded automatically at warm-up time — no need to set `models`.
  For unsupported languages, use the `models` parameter to configure a custom model.
  See [Presidio supported languages](https://microsoft.github.io/presidio/analyzer/languages/).
- **entities** (<code>list\[str\] | None</code>) – List of PII entity types to detect and anonymize (e.g. `["PERSON", "EMAIL_ADDRESS"]`).
  If `None`, all supported entity types are used.
  See [Presidio supported entities](https://microsoft.github.io/presidio/supported_entities/).
- **score_threshold** (<code>float</code>) – Minimum confidence score (0-1) for a detected entity to be anonymized. Defaults to `0.35`.
  See [Presidio analyzer documentation](https://microsoft.github.io/presidio/analyzer/).
- **models** (<code>list\[dict\[str, str\]\] | None</code>) – Advanced override: list of spaCy model configurations.
  Each entry must contain `"lang_code"` and `"model_name"` keys,
  e.g. `[{"lang_code": "fr", "model_name": "fr_core_news_md"}]`.
  Use this only when you need a specific model variant or a language not covered by the
  built-in mapping. If `None`, the model is selected automatically from `SPACY_DEFAULT_MODELS`
  based on `language`.

#### warm_up

```python
warm_up() -> None
```

Initializes the Presidio analyzer and anonymizer engines.

This method loads the underlying NLP models. In a Haystack Pipeline,
this is called automatically before the first run.

#### run

```python
run(documents: list[Document]) -> dict[str, list[Document]]
```

Anonymizes PII in the provided Documents.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – List of Documents whose text content will be anonymized.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with key `documents` containing the cleaned Documents.

## haystack_integrations.components.preprocessors.presidio.presidio_text_cleaner

### PresidioTextCleaner

Anonymizes PII in plain strings using [Microsoft Presidio](https://microsoft.github.io/presidio/).

Accepts a list of strings, detects personally identifiable information (PII), and returns
a new list of strings with PII replaced by entity type placeholders (e.g. `<PERSON>`).
Useful for sanitizing user queries before they are sent to an LLM.

The analyzer and anonymizer engines are loaded on the first call to `run()`,
or by calling `warm_up()` explicitly beforehand.

### Usage example

```python
from haystack_integrations.components.preprocessors.presidio import PresidioTextCleaner

cleaner = PresidioTextCleaner()
result = cleaner.run(texts=["Hi, I am John Smith, call me at 212-555-1234"])
print(result["texts"][0])
# Hi, I am <PERSON>, call me at <PHONE_NUMBER>
```

#### SPACY_DEFAULT_MODELS

```python
SPACY_DEFAULT_MODELS: dict[str, str] = _SPACY_DEFAULT_MODELS
```

Mapping from ISO 639-1 language code to the largest available spaCy model for that language.

Used to automatically select an NLP model when `models` is not specified.
See [spaCy documentation](https://spacy.io/models) for the full list of available spaCy models.

#### __init__

```python
__init__(
    *,
    language: str = "en",
    entities: list[str] | None = None,
    score_threshold: float = 0.35,
    models: list[dict[str, str]] | None = None
) -> None
```

Initializes the PresidioTextCleaner.

**Parameters:**

- **language** (<code>str</code>) – ISO 639-1 language code for PII detection. Defaults to `"en"`.
  For languages in the built-in mapping (e.g. `"de"`, `"fr"`, `"es"`), the appropriate
  spaCy model is loaded automatically at warm-up time — no need to set `models`.
  For unsupported languages, use the `models` parameter to configure a custom model.
  See [Presidio supported languages](https://microsoft.github.io/presidio/analyzer/languages/).
- **entities** (<code>list\[str\] | None</code>) – List of PII entity types to detect and anonymize (e.g. `["PERSON", "PHONE_NUMBER"]`).
  If `None`, all supported entity types are used.
  See [Presidio supported entities](https://microsoft.github.io/presidio/supported_entities/).
- **score_threshold** (<code>float</code>) – Minimum confidence score (0-1) for a detected entity to be anonymized. Defaults to `0.35`.
  See [Presidio analyzer documentation](https://microsoft.github.io/presidio/analyzer/).
- **models** (<code>list\[dict\[str, str\]\] | None</code>) – Advanced override: list of spaCy model configurations.
  Each entry must contain `"lang_code"` and `"model_name"` keys,
  e.g. `[{"lang_code": "fr", "model_name": "fr_core_news_md"}]`.
  Use this only when you need a specific model variant or a language not covered by the
  built-in mapping. If `None`, the model is selected automatically from `SPACY_DEFAULT_MODELS`
  based on `language`.

#### warm_up

```python
warm_up() -> None
```

Initializes the Presidio analyzer and anonymizer engines.

This method loads the underlying NLP models. In a Haystack Pipeline,
this is called automatically before the first run.

#### run

```python
run(texts: list[str]) -> dict[str, list[str]]
```

Anonymizes PII in the provided strings.

**Parameters:**

- **texts** (<code>list\[str\]</code>) – List of strings to anonymize.

**Returns:**

- <code>dict\[str, list\[str\]\]</code> – A dictionary with key `texts` containing the cleaned strings.

## haystack_integrations.hooks.presidio.presidio_pii_hook

### PresidioPIIHook

Detects personally identifiable information (PII) in an Agent's messages with Microsoft Presidio.

Register the same instance at `before_llm`, so PII never reaches the LLM provider, and at `after_run`, so the final
reply returned to the user is checked too. The final reply is produced by the last LLM call, so no `before_llm`
hook runs after it:

<!-- test-ignore -->

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_integrations.hooks.presidio import PresidioPIIHook

pii_hook = PresidioPIIHook(
    entity_score_thresholds={"PERSON": 0.6},
    entity_actions={"CREDIT_CARD": "block", "IBAN_CODE": "stop"},
)
agent = Agent(
    chat_generator=OpenAIChatGenerator(),
    hooks={"before_llm": [pii_hook], "after_run": [pii_hook]},
)
result = agent.run(messages=[ChatMessage.from_user("Hi, I'm John Smith from Berlin.")])
# The LLM receives: "Hi, I'm <PERSON> from <LOCATION>."
```

Each run checks the messages it has not checked yet: user and assistant text and tool results. System messages,
system reminders, tool-call arguments, reasoning, images, and files are not checked. Every detected entity maps to
an action, and a message takes the strictest action among its entities:

- `"redact"` (default): Presidio's anonymizer replaces each entity with a placeholder such as `<PERSON>`.
- `"block"`: The message's text is replaced with `block_message`. Mid-run, a system reminder is added telling the
  model the message was removed; once the run is exiting, the reply is replaced without a reminder.
- `"stop"`: The message is redacted and the run stops before the next LLM call, with `exit_reason` set to
  `"pii_detected"`.

Hooks at the same hook point run in list order, and any hook listed before this one sees unredacted messages.
List it first, for example ahead of a `CompactionHook` that summarizes the conversation with an LLM. To keep PII
out of a `ToolResultOffloadHook` store, also register it first at `after_tool`:

<!-- test-ignore -->

```python
hooks = {
    "after_tool": [pii_hook, offload_hook],
    "before_llm": [pii_hook, compaction_hook],
    "after_run": [pii_hook],
}
```

Limitations:

- When the Agent or its chat generator has a `streaming_callback`, the reply streams to the user before any hook
  runs, so PII in the reply cannot be stopped in that mode.
- Because the model only sees placeholders, the tool-call arguments it writes contain placeholders too, so a tool
  that needs the real value (for example, an email address to send to) receives `<EMAIL_ADDRESS>`.
- If the hook runs at `after_tool` and blocks the result of a tool in `exit_conditions`, the reminder becomes the
  Agent's last message. Read the tool result from `result["messages"]` instead of `result["last_message"]`.
- After an `on_exit` hook keeps the run going, messages blocked later in the run get no reminder.

Each run records its decision on the hook's tracing span under `haystack.agent.hook.pii.*`: how many messages it
checked, whether anything was found, the action taken, the count of each entity type, and whether Presidio failed.
The checked and rewritten texts are recorded as content tags, so they are only emitted when content tracing is on.
With content tracing on, traces contain raw PII: this hook's input tag holds the original text, and the Agent's own
spans record messages before any hook redacts them.

#### SPACY_DEFAULT_MODELS

```python
SPACY_DEFAULT_MODELS: dict[str, str] = _SPACY_DEFAULT_MODELS
```

Mapping from ISO 639-1 language code to the largest available spaCy model for that language.

Used to automatically select an NLP model when `models` is not specified.
See [spaCy documentation](https://spacy.io/models) for the full list of available spaCy models.

#### __init__

```python
__init__(
    *,
    language: str = "en",
    entities: list[str] | None = None,
    score_threshold: float = 0.35,
    entity_score_thresholds: dict[str, float] | None = None,
    entity_actions: dict[str, PIIAction] | None = None,
    on_error: PIIErrorAction = "block",
    block_message: str = _DEFAULT_BLOCK_MESSAGE,
    models: list[dict[str, str]] | None = None
) -> None
```

Initialize the hook.

**Parameters:**

- **language** (<code>str</code>) – ISO 639-1 language code for PII detection. Defaults to `"en"`.
  For languages in `SPACY_DEFAULT_MODELS`, the spaCy model is loaded automatically at warm-up time.
  See [Presidio supported languages](https://microsoft.github.io/presidio/analyzer/languages/).
- **entities** (<code>list\[str\] | None</code>) – PII entity types to detect, for example `["PERSON", "PHONE_NUMBER"]`.
  If `None`, all entity types Presidio supports for `language` are detected.
  See [Presidio supported entities](https://microsoft.github.io/presidio/supported_entities/).
- **score_threshold** (<code>float</code>) – Minimum confidence score (0-1) for a detected entity to count, for entity types not
  listed in `entity_score_thresholds`. Defaults to `0.35`.
- **entity_score_thresholds** (<code>dict\[str, float\] | None</code>) – Per-entity-type minimum confidence scores (0-1) that override
  `score_threshold`, for example `{"PERSON": 0.6}` to ignore low-confidence name matches.
- **entity_actions** (<code>dict\[str, PIIAction\] | None</code>) – Per-entity-type actions, one of `"redact"`, `"block"`, or `"stop"`, for example
  `{"CREDIT_CARD": "block"}`. Entity types not listed are redacted.
- **on_error** (<code>PIIErrorAction</code>) – What to do with a message when Presidio raises an error while checking it:
  `"block"` (default) replaces it with `block_message`, `"stop"` also stops the run, and `"pass"` leaves
  the message unchanged, which can let PII through.
- **block_message** (<code>str</code>) – The text that replaces the text of a blocked message.
- **models** (<code>list\[dict\[str, str\]\] | None</code>) – Advanced override: spaCy model configurations, each with `"lang_code"` and `"model_name"`
  keys, for example `[{"lang_code": "fr", "model_name": "fr_core_news_md"}]`. If `None`, the model is
  selected from `SPACY_DEFAULT_MODELS` based on `language`.

**Raises:**

- <code>ValueError</code> – If a score threshold is outside 0-1, or an action or `on_error` value is unknown.

#### warm_up

```python
warm_up() -> None
```

Load the Presidio analyzer and anonymizer engines and check the configured entity types.

The Agent calls this from its own `warm_up`. Calling it again does nothing.

**Raises:**

- <code>ValueError</code> – If no spaCy model is available for `language`, or an entity type in `entities`,
  `entity_score_thresholds`, or `entity_actions` is not supported for `language`, or an entity type in
  `entity_score_thresholds` or `entity_actions` is not in `entities`.

#### run

```python
run(state: State) -> None
```

Check the messages not checked yet and redact, block, or stop as configured.

**Parameters:**

- **state** (<code>State</code>) – The Agent's live `State`. Its `messages` are rewritten in place; `stop_run` is set when a message
  triggers `"stop"`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the hook to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized hook.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> PresidioPIIHook
```

Deserialize the hook from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary produced by `to_dict`.

**Returns:**

- <code>PresidioPIIHook</code> – The deserialized hook.
