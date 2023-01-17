- Title: TableSpan Dataclass
- Decision driver: Sebastian Lee
- Start Date: 2023-01-17
- Proposal PR: (fill in after opening the PR)
- Github Issue: https://github.com/deepset-ai/haystack/issues/3616

# Summary

When returning answers for a TableQA pipeline we would like to return the column and row index as the answer location
within the table since the table is either returned as a list of lists or a pandas dataframe in Haystack.
This would allow users to easily look up the answer in the returned table to fetch the text directly from the table,
identify the row or column labels for that answer, or generally perform operations on the table near or around the
answer cell.

# Basic example <a name="#basic-example"></a>

When applicable, write a snippet of code showing how the new feature would be used.
```python
import pandas as pd
from haystack.nodes import TableReader
from haystack import Document

data = {
    "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
    "age": ["58", "47", "60"],
    "number of movies": ["87", "53", "69"],
    "date of birth": ["18 december 1963", "11 november 1974", "6 may 1961"],
}
table_doc = Document(content=pd.DataFrame(data), content_type="table")
reader = TableReader(model_name_or_path="google/tapas-base-finetuned-wtq", max_seq_len=128)
prediction = reader.predict(query="Who was in the most number of movies?", documents=[table_doc])
answer = prediction["answers"][0]

# New feature
print(answer.context.iloc[answer.offsets_in_context[0].col, answer.offsets_in_context[0].row])
```

# Motivation

## Why do we need this feature?
To allow users to easily look up the answer cell in the returned table to fetch the answer text
directly from the table, identify the row or column labels for that answer, or generally perform operations on the table
near or around the answer cell.

Currently, we return the location of the answer in the **linearized** version of the table, so we can use the
`Span` dataclass. The `Span` dataclass is reproduced below:
```python
@dataclass
class Span:
    start: int
    end: int
    """
    Defining a sequence of characters (Text span) or cells (Table span) via start and end index.
    For extractive QA: Character where answer starts/ends
    For TableQA: Cell where the answer starts/ends (counted from top left to bottom right of table)

    :param start: Position where the span starts
    :param end:  Position where the span ends
    """
```
This is inconvenient for users because they would need to know how the table is linearized (column major or row major)
so they could reconstruct the column and row indices of the answer before they could locate the answer cell in the table.

## What use cases does it support?
Some examples are already stated above but to recap, to easily perform operations on the table near or around the answer cell.

## What's the expected outcome?
The addition of a new dataclass called `TableSpan` that would look like
```python
@dataclass
class TableSpan:
    col: int
    row: int
    """
    Defining a table cell via the column and row index.

    :param col: Column index of the span
    :param row: Row index of the span
    """
```
**Note:** I am open to a name change since this isn't really a span, but the location of a single table cell.

# Detailed design

**New terminology:** `TableSpan` or something similar (e.g. `Cell`,`CellLoc`, etc.). The new name for the dataclass to
store the column and row index of the answer cell.

**Basic Example:** [Above Basic Example](#basic-example)

## Code changes
- Addition of `TableSpan` dataclass to https://github.com/deepset-ai/haystack/blob/main/haystack/schema.py
```python
@dataclass
class TableSpan:
    col: int
    row: int
    """
    Defining a table cell via the column and row index.

    :param col: Column index of the span
    :param row: Row index of the span
    """
```

- Updating code (e.g. schema objects, classes, functions) that use `Span` to also support `TableSpan` where appropriate.
This includes:
- Updating the `Answer` dataclass to support `TableSpan` as a valid type for `offsets_in_document` and `offsets_in_context`
```python
@dataclass
 class Answer:
     answer: str
     type: Literal["generative", "extractive", "other"] = "extractive"
     score: Optional[float] = None
     context: Optional[Union[str, pd.DataFrame]] = None
     offsets_in_document: Optional[List[Span], List[TableSpan]] = None
     offsets_in_context: Optional[List[Span], List[TableSpan]] = None
     document_id: Optional[str] = None
     meta: Optional[Dict[str, Any]] = None
```
- Updating any functions that accept table answers as input to use the new `col` and `row` variables instead of `start` and `end` variables.
This type of check for table answers is most likely already done by checking if the `context` is of type `pd.DataFrame`.
- `TableReader` and `RCIReader` to return `TableSpan` objects instead of `Span`

## Edge Case/Bug
Internally, Haystack stores a table as a pandas DataFrame in the `Answer` dataclass, which does not treat the column
labels as the first row in the table.
However, in Haystack's rest-api the table is converted into a list of lists format where the column labels are
stored as the first row, which can be seen [here](https://github.com/deepset-ai/haystack/pull/3872), which is consistent
with the `Document.to_dict()` method seen [here](https://github.com/deepset-ai/haystack/blob/6af4f14fe0d375a1ae0ced18930a9239401231c7/haystack/schema.py#L164-L165).

This means that the current `Span` and (new) `TableSpan` dataclass point to the wrong location when the table is
converted to a list of lists.

For example, the following code
```python
import pandas as pd
from haystack import Document

data = {
    "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
    "age": ["58", "47", "60"],
    "number of movies": ["87", "53", "69"],
    "date of birth": ["18 december 1963", "11 november 1974", "6 may 1961"],
}
table_doc = Document(content=pd.DataFrame(data), content_type="table")
span = (0, 0)
print(table_doc.content.iloc[span])  # prints "brad pitt"

dict_table_doc = table_doc.to_dict()
print(dict_table_doc["content"][span[0]][span[1]])  # prints "actors"
```

I see a few ways to address this:
- One way to address this would be to update the `Span` (or `TableSpan`) to be updated when the table is converted into a list of lists.
- Another way would be to only store the table internally as a list of lists.

# Drawbacks

Look at the feature from the other side: what are the reasons why we should _not_ work on it? Consider the following:

- What's the implementation cost, both in terms of code size and complexity?
- Can the solution you're proposing be implemented as a separate package, outside of Haystack?
- Does it teach people more about Haystack?
- How does this feature integrate with other existing and planned features?
- What's the cost of migrating existing Haystack pipelines (is it a breaking change?)?

There are tradeoffs to choosing any path. Attempt to identify them here.

# Alternatives

## What's the impact of not adding this feature?
Requiring users to figure out how to interpret the linearized answer cell coordinates to reconstruct the row and column indices
to be able to access the answer cell in the returned tabel.

## Other designs
1. Expand `Span` dataclass to have optional `col` and `row` fields. This would require a similar check as `TableSpan`, but instead
require checking for which of the elements are populated, which seems unnecessarily complex.
```python
@dataclass
class Span:
    start: int = None
    end: int = None
    col: int = None
    row: int = None
```
2. Use the existing `Span` dataclass and put the row index and column index as the `start` and `end` respectively.
This may be confusing to users since it is not obvious that `start` should refer to `row` and `end` should refer to `column`.
```python
answer_cell_offset = Span(start=row_idx, end=col_idx)
```
3. Provide a convenience function shown [here](https://github.com/deepset-ai/haystack/issues/3616#issuecomment-1361300067)
to help users convert the linearized `Span` back to row and column indices. I believe this solution is non-ideal since it would
require a user of the rest_api to access a python function to convert the linearized indices back into row and column indices.

# Adoption strategy

If we implement this proposal, how will the existing Haystack users adopt it? Is
this a breaking change? Can we write a migration script?

# How we teach this

Would implementing this feature mean the documentation must be re-organized
or updated? Does it change how Haystack is taught to new developers at any level?

How should this feature be taught to the existing Haystack users (for example with a page in the docs,
a tutorial, ...).

# Unresolved questions

Optional, but suggested for first drafts. What parts of the design are still
TBD?
