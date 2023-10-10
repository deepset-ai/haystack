- Title: Document Stores filter specification for Haystack 2.x
- Decision driver: Silvano Cerza
- Start Date: 2023-10-08
- Proposal PR: [#6001](https://github.com/deepset-ai/haystack/pull/6001)

# Summary

This proposal introduces a new fully detailed and extensible specification for filtering Document Stores in Haystack 2.x.
This comes from my personal experience and struggles trying to implement filters conversion for `ElasticsearchDocumentStore`.

# Basic example

```
{
  "conditions": [
    { "field": "age", "operator": ">=", "value": 18 },
    {
      "operator": "OR",
      "conditions": [
        { "field": "country", "operator": "==", "value": "USA" },
        { "field": "country", "operator": "==", "value": "Canada" }
      ]
    }
  ]
}
```

The above snippet would be equivalent to the following Python code:

```python
age >= 18 and (country == "USA" or country == "Canada)
```

# Motivation

Filtering in Haystack 1.x has no detailed clear specification, we only have an [high level overview][filters-high-level-doc] in the documentation that also mixes REST APIs documentation in. It's also inherited from MongoDB and is a subset of the Mongo Query Language.

Converting from the current filters to another query language is quite hard as there are tons of corner cases. Handling nested filters is usually really error prone as the operators can be keys, this requires ton of nested and/or recursive logic to figure out whether the current key is a field or an operator. There's also quite some backtracking involved as the field could be two or three levels above its comparison operator and/or value.

As a practical example the below two filters are equivalent. Given that they're structured differently and `$and` is implicit in the second one:

```
{"number": {"$and": [{"$lte": 2}, {"$gte": 0}]}}

{"number": {"$lte": 2, "$gte": 0}}
```

With the newly proposed approach both filters would be equivalent to:

```
{
    "operator": "AND",
    "conditions": [
        { "field": "number", "operator": "<=", "value": 2 },
        { "field": "number", "operator": ">=", "value": 0 },
    ]
}
```

As you can see all the required information is one the same level and clearly recognisable. This makes it much easier both to read by a human and convert with code.

In Python code:

```python
number <= 2 AND number >= 0
```

# Detailed design

Filters top level must be a dictionary.

There are two types of dictionaries:

- Comparison
- Logic

Top level can be either be a Comparison or Logic dictionary.

Comparison dictionaries must contain the keys:

- `field`
- `operator`
- `value`

Logic dictionaries must contain the keys:

- `operator`
- `conditions`

`conditions` key must be a list of dictionaries, either Comparison or Logic.

`operator` values in Comparison dictionaries must be:

- `==`
- `!=`
- `>`
- `>=`
- `<`
- `<=`
- `in`
- `not in`

`operator` values in Logic dictionaries must be:

- `NOT`
- `OR`
- `AND`

---

As an example this:

```
{
    "$and": {
        "type": {"$eq": "article"},
        "$or": {"genre": {"$in": ["economy", "politics"]}, "publisher": {"$eq": "nytimes"}},
        "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
        "rating": {"$gte": 3},
    }
}
```

would convert to:

```
{
    "operator": "AND",
    "conditions": [
        { "field": "type", "operator": "==", "value": "article" },
        {
            "operator": "OR",
            "conditions": [
                { "field": "genre", "operator": "in", "value": ["economy", "politics"] },
                { "field": "publisher", "operator": "==", "value": "nytimes" },
            ]
        },
        { "field": "date", "operator": ">=", "value": "2015-01-01" },
        { "field": "date", "operator": "<", "value": "2021-01-01" },
        { "field": "rating", "operator": ">=", "value": 3 },
    ]
}
```

In Python code:

```python
type == "article" and (
    genre in ["economy", "politics"] or publisher == "nytimes"
) and date >= "2015-01-01" and date < "2021-01-01" and rating >= 3
```

Dates have been kept as strings but ideally in the new implementation they would be converted to `datetime` instances so the Document Store will be able to convert it to whatever format it needs to actually compare them. As different Document Stores might have different ways of storing the same value it's important that they handle the conversion from Python type to stored type.

Another thing that in my opinion should be changed is the that filtering metadata fields must be explicitly specified for filtering. In the example above all `field`s would be prefixed with `metadata.` to get in return the expected `Document`s. e.g. `date` -> `metadata.date`
This connects to `Document` implementation and is not the focus of this proposal, but it should be taken into account.

# Drawbacks

The only drawback would be that we need to adapt the existing Document Stores already created for Haystack 2.x to support this filtering system. `MemoryDocumentStore`, `ElasticsearchDocumentStore`, `ChromaDocumentStore` and `MarqoDocumentStore` are the currently existing Document Stores.

# Alternatives

An alternative would be keeping the current strategy of declaring filters.
This wouldn't require any change but supporting a new filtering language after the release of Haystack 2.x would be more difficult than doing it now.

# Adoption strategy

We're going to release this new strategy of filters declaration for Haystack 2.x. At the same time we'll deprecate the current strategy but we'll keep supporting it for a while.

Since we're going to provide an utility function to convert from old style to new style it will be easy for Document Stores to support both.

# How we teach this

We're going to provide documentation and specifications on how the filters should be declared, this proposal is a good starting point as it already defines the specs.

We're also going to provide utility functions to migrate filters from old style to new style.

# Unresolved questions

This is the full design and there are no unresolved questions.

[filters-high-level-doc]: https://docs.haystack.deepset.ai/docs/metadata-filtering
