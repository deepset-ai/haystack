- Title: Document class for Haystack 2.0
- Decision driver: ZanSara
- Start Date: 2023-09-07
- Proposal PR: 5738

# Summary

With Haystack 2.0 we want to provide a lot more flexibility to Pipelines and Components. In a lot of situations,
we found that the Document class inherited from Haystack 1.x was not up to the task: therefore we chose to expand its
API to work best in this new paradigm.

# Basic example

Documents 2.0 have two fundamental differences with Documents 1.x:

- They have more than one content field. Documents 1.x only have a `content: Any` field that needs to match with the
    `content_type` field in meaning. Documents 2.0 instead support `text`, `array`, `dataframe` and `blob`, each typed
    correctly.

- The `content_type` field is gone: In Haystack 1.x we used the `content_type` field to interpret the data contained
    in the `content` field: with the new design, this won't be necessary any longer. Haystack 2.0, however, have
    a `mime_type` field that helps interpret the content of the `blob` field if necessary.

# Motivation

During the development of Haystack 2.0 components, we often found ourselves hold back by the design limitations of
the Document class. Unlike in Haystack 1.x, Documents now carry more information across the pipeline: for example,
they might contain the file they originated from, they might support more datatypes, etc.

Therefore we decided to extend the Document class to support a wider array of data.

# Detailed design

The design of this class was inspired by the [DocArray API](https://docarray.jina.ai/fundamentals/document/).

Here is the high-level API of the new Document class:

```python
@dataclass(frozen=True)
class Document:
    id: str = field(default_factory=str)
    text: Optional[str] = field(default=None)
    array: Optional[numpy.ndarray] = field(default=None)
    dataframe: Optional[pandas.DataFrame] = field(default=None)
    blob: Optional[bytes] = field(default=None)
    mime_type: str = field(default="text/plain")
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False)
    id_hash_keys: List[str] = field(default_factory=lambda: ["text", "array", "dataframe", "blob"], hash=False)
    score: Optional[float] = field(default=None, compare=True)
    embedding: Optional[numpy.ndarray] = field(default=None, repr=False)

    def to_dict(self):
        """
        Saves the Document into a dictionary.
        """

    def to_json(self, json_encoder: Optional[Type[DocumentEncoder]] = None, **json_kwargs):
        """
        Saves the Document into a JSON string that can be later loaded back. Drops all binary data from the blob field.
        """

    @classmethod
    def from_dict(cls, dictionary):
        """
        Creates a new Document object from a dictionary of its fields.
        """

    @classmethod
    def from_json(cls, data, json_decoder: Optional[Type[DocumentDecoder]] = None, **json_kwargs):
        """
        Creates a new Document object from a JSON string.
        """

    def flatten(self) -> Dict[str, Any]:
        """
        Returns a dictionary with all the document fields and metadata on the same level.
        Helpful for filtering in document stores.
        """

```

As you can notice, the main difference is the management of the content fields: we now have:

- `text`: for text data
- `array`: for array-like data, for example images, audio, video
- `dataframe`: for tabular data
- `blob`: for binary data.

In order to help interpret the content of these field, there's a `mime_type` field that components can use to figure out
how to use the content fields they need.

There are additional information that we may want to add, for example `path`. For now such info can be
kept into the metadata: if we realize we access it extremely often while processing Documents we should consider
bringing those fields out of `metadata` as top-level properties of the dataclass.


# Drawbacks

As the Document class becomes a bit more complex, components need to be adapted to it. This may cause some issues
to DocumentStores, because now they not only need to be able to store text but binary blobs as well.

We can imagine that some very simple DocumentStore will refuse to store the binary blobs. Fully-featured,
production-ready document stores instead should be able to find a way to store such blobs.


# Unresolved questions

Are the 4 content fields appropriate? Are there other content types we can consider adding?
