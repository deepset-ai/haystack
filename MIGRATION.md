# Migration Guide

This document is meant to provide a guide for migrating from Haystack v2.X to v3.0.

---

## How to Document a Breaking Change

When you merge a breaking change into the v3 branch, add an entry to this file under the appropriate section below.
Follow this structure:

### Entry template

```markdown
### <Short title describing what changed>

**What changed:** One or two sentences describing the change — what was removed, renamed, or altered.

**Why:** Brief motivation (e.g. simplification, API consistency, dependency reduction).

**How to migrate:**

Before (v2.x):
\`\`\`python
# example using the old API
from haystack.components.foo import OldComponent
component = OldComponent(old_param="value")
\`\`\`

After (v3.0):
\`\`\`python
# example using the new API
from haystack.components.foo import NewComponent
component = NewComponent(new_param="value")
\`\`\`
```

### Tips

- **One entry per breaking change.** Don't bundle unrelated changes into a single entry.
- **Include a working code example** for every rename, removal, or signature change.
- **Link to the PR** when extra context would help (e.g. `See [#1234](https://github.com/deepset-ai/haystack/pull/1234)`).

---

## Breaking Changes

<!-- Add entries here as v3 development progresses. Example below shows the expected format. -->

### Example entry: `Document.dataframe` field removed

**What changed:** The `dataframe` field on `Document` and the `ExtractedTableAnswer` dataclass have been removed. `pandas` is no longer a required dependency.

**Why:** Reduces the default installation footprint. Components that need `pandas` will raise an informative error prompting the user to install it explicitly.

**How to migrate:**

Before (v2.x):
```python
from haystack.dataclasses import Document
import pandas as pd

doc = Document(content=pd.DataFrame({"col": [1, 2, 3]}))
```

After (v3.0):
```python
# Store tabular data as plain content or create a custom component that returns pandas DataFrames as needed.
from haystack.dataclasses import Document

doc = Document(content="col\n1\n2\n3")
```
