---
title: "Preprocessors"
id: experimental-preprocessors-api
description: "Pipelines wrapped as components."
slug: "/experimental-preprocessors-api"
---

<a id="haystack_experimental.components.preprocessors.md_header_level_inferrer"></a>

## Module haystack\_experimental.components.preprocessors.md\_header\_level\_inferrer

<a id="haystack_experimental.components.preprocessors.md_header_level_inferrer.MarkdownHeaderLevelInferrer"></a>

### MarkdownHeaderLevelInferrer

Infers and rewrites header levels in Markdown text to normalize hierarchy.

    First header → Always becomes level 1 (#)
    Subsequent headers → Level increases if no content between headers, stays same if content exists
    Maximum level → Capped at 6 (######)

    ### Usage example
    ```python
    from haystack import Document
    from haystack_experimental.components.preprocessors import MarkdownHeaderLevelInferrer

    # Create a document with uniform header levels
    text = "## Title
## Subheader
Section
## Subheader
More Content"
    doc = Document(content=text)

    # Initialize the inferrer and process the document
    inferrer = MarkdownHeaderLevelInferrer()
    result = inferrer.run([doc])

    # The headers are now normalized with proper hierarchy
    print(result["documents"][0].content)
    > # Title
## Subheader
Section
## Subheader
More Content
    ```

<a id="haystack_experimental.components.preprocessors.md_header_level_inferrer.MarkdownHeaderLevelInferrer.__init__"></a>

#### MarkdownHeaderLevelInferrer.\_\_init\_\_

```python
def __init__()
```

Initializes the MarkdownHeaderLevelInferrer.

<a id="haystack_experimental.components.preprocessors.md_header_level_inferrer.MarkdownHeaderLevelInferrer.run"></a>

#### MarkdownHeaderLevelInferrer.run

```python
@component.output_types(documents=list[Document])
def run(documents: list[Document]) -> dict
```

Infers and rewrites the header levels in the content for documents that use uniform header levels.

**Arguments**:

- `documents`: list of Document objects to process.

**Returns**:

dict: a dictionary with the key 'documents' containing the processed Document objects.

