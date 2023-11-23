- Title: TableCell Dataclass
- Decision driver: Sebastian Lee
- Start Date: 2023-01-17
- Proposal PR: https://github.com/deepset-ai/haystack/pull/3875
- Github Issue: https://github.com/deepset-ai/haystack/issues/3616

# Summary

When returning answers for a TableQA pipeline we would like to return the column and row index as the answer location
within the table since the table is either returned as a list of lists in Haystack.
This would allow users to easily look up the answer in the returned table to fetch the text directly from the table,
identify the row or column labels for that answer, or generally perform operations on the table near or around the
answer cell.

# Basic Example

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
# answer.context -> [["actor", "age", "number of movies"], ["Brad Pitt",...], [...]]
# answer.offsets_in_context[0] -> (row=1, col=1)
print(answer.context[answer.offsets_in_context[0].row][answer.offsets_in_context[0].col])
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
The addition of a new dataclass called `TableCell` that would look like
```python
@dataclass
class TableCell:
    row: int
    col: int
    """
    Defining a table cell via the row and column index.

    :param row: Row index of the cell
    :param col: Column index of the cell
    """
```

# Detailed design

**New terminology:** `TableCell`, the new name for the dataclass to
store the column and row index of the answer cell.

**Basic Example:** [Above Basic Example](#basic-example)

## Code changes
- Addition of `TableCell` dataclass to https://github.com/deepset-ai/haystack/blob/main/haystack/schema.py
```python
@dataclass
class TableCell:
    row: int
    col: int
    """
    Defining a table cell via the row and column index.

    :param row: Row index of the cell
    :param col: Column index of the cell
    """
```

- Updating code (e.g. schema objects, classes, functions) that use `Span` to also support `TableCell` where appropriate.
This includes:
- Updating the `Answer` dataclass to support `TableCell` as a valid type for `offsets_in_document` and `offsets_in_context`
```python
@dataclass
 class Answer:
     answer: str
     type: Literal["generative", "extractive", "other"] = "extractive"
     score: Optional[float] = None
     context: Optional[Union[str, List[List]]] = None
     offsets_in_document: Optional[List[Span], List[TableCell]] = None
     offsets_in_context: Optional[List[Span], List[TableCell]] = None
     document_id: Optional[str] = None
     meta: Optional[Dict[str, Any]] = None
```
- Similar to how we can return a list of `Span`s, we would allow a list of `TableCell`s to be returned to handle the case
 when multiple `TableCell`s are returned to form a final answer.
- Updating any functions that accept table answers as input to use the new `col` and `row` variables instead of `start` and `end` variables.
This type of check for table answers is most likely already done by checking if the `context` is of type `pd.DataFrame`.
- `TableReader` and `RCIReader` to return `TableCell` objects instead of `Span`.

Changes related to the Edge Case/Bug below
- Update `Document.content` and `Answer.context` to use `List[List]` instead of `pd.DataFrame`.
- Update `TableReader` nodes to convert table from `List[List]` into `pd.DataFrame` before inputting to the model.

## Edge Case/Bug
Internally, Haystack stores a table as a pandas DataFrame in the `Answer` dataclass, which does not treat the column
labels as the first row in the table.
However, in Haystack's rest-api the table is converted into a list of lists format where the column labels are
stored as the first row, which can be seen [here](https://github.com/deepset-ai/haystack/pull/3872), which is consistent
with the `Document.to_dict()` method seen [here](https://github.com/deepset-ai/haystack/blob/6af4f14fe0d375a1ae0ced18930a9239401231c7/haystack/schema.py#L164-L165).

This means that the current `Span` and (new) `TableCell` dataclass point to the wrong location when the table is
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

We have decided to store the table internally as a list of lists to avoid this issue. See discussion starting [here](https://github.com/deepset-ai/haystack/pull/3875#discussion_r1088766318).

# Drawbacks

Look at the feature from the other side: what are the reasons why we should _not_ work on it? Consider the following:

- What's the implementation cost, both in terms of code size and complexity?

I don't believe this will require too much code change since we already check for Table like answers by checking if the
returned context is of type string or pandas Dataframe.

- Can the solution you're proposing be implemented as a separate package, outside of Haystack?

Technically yes, but since it affects core classes like `TableReader`, and `RCIReader` it makes sense to implement in
Haystack.

- Does it teach people more about Haystack?

It would update already existing documentation and tutorials of Haystack.

- How does this feature integrate with other existing and planned features?

This feature directly integrates and impacts the TableQA feature of Haystack.

- What's the cost of migrating existing Haystack pipelines (is it a breaking change?)?

Yes there are breaking changes that would affect end users.
1. The way to access the offsets in returned Answers would be different.
Following the deprecation policy we will support both `Span` and `TableCell` (can be toggled between using a boolean flag)
for 2 additional versions of Haystack.
2. Tables in Haystack Documents and Answers will change from type pandas Dataframe to a list of lists.

# Alternatives

## What's the impact of not adding this feature?
Requiring users to figure out how to interpret the linearized answer cell coordinates to reconstruct the row and column indices
to be able to access the answer cell in the returned table.

## Other designs
1. Expand `Span` dataclass to have optional `col` and `row` fields. This would require a similar check as `TableCell`, but instead
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

## How will the existing Haystack users adopt it?
Haystack users would immediately experience this change once they update their installation of Haystack if they were using
the TableQA reader. This would be a breaking change since it would change the `offsets_in_document` and
`offsets_in_context` in the returned `Answer`. I'm not sure if there would be a straightforward way to write a migration
script for this change.

# How we teach this

Would implementing this feature mean the documentation must be re-organized
or updated? Does it change how Haystack is taught to new developers at any level?

- The API docs for `TableCell` would need to be added.
- The documentation page for [Table Question Answering](https://docs.haystack.deepset.ai/docs/table_qa) would need to be updated.
- Update the (TableQa tutorial)[https://github.com/deepset-ai/haystack-tutorials/blob/main/tutorials/15_TableQA.ipynb]
to reflect the `Span` is no longer linearzied.

# Unresolved questions

No more unresolved questions.
