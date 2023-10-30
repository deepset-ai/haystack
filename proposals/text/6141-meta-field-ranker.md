- Title: Meta Field Ranker
- Decision driver: @domenicocinque
- Start Date: 2023-10-20
- Proposal PR: https://github.com/deepset-ai/haystack/pull/6141
- Github Issue or Discussion: https://github.com/deepset-ai/haystack/issues/6054

# Summary

This ranker should allow to sort the documents based on a meta field of choice.

# Basic example

The ranker could be instantiated as follows:

``` python
ranker = MetaFieldRanker(
    meta_field="rating",
    weight="0.5",
    ascending=False,
    top_k=3,
)
```
In the context of a simple pipeline with a retriever and a MetaFieldRanker in which
the documents are provided with a meta field "rating". The documents are first retrieved by the retriever and
then sorted by the MetaFieldRanker.

``` python
pipeline = Pipeline()
pipeline.add_component(component=InMemoryBM25Retriever(document_store=document_store, top_k=20)
, name="Retriever")
pipeline.add_component(component=MetaFieldRanker(meta_field="rating"), name="Ranker")
pipeline.connect("Retriever.documents", "MetaFieldRanker.documents")
```

# Motivation

I found the need for this feature while working on system that retrieves books based on their description and the
similarity to the query. After retrieving the documents it makes sense to expose them to the user in order of popularity.
This is just one example of a use case for this feature, but I think it could be useful in many other contexts.

# Detailed design

The actual implementation of the ranker is very similar to the already present RecentnessRanker. The main difference
is to remove the date parsing logic.

# Drawbacks

The main drawback is that it would be very similar to the already present RecentnessRanker. However, this could be
solved by making the RecentnessRanker a subclass of the MetaFieldRanker and adding the date parsing logic to it.
Apart from that, it is a very simple component that should not have any other drawbacks.

# Alternatives

The alternative is to make the user implement its own ranking logic.

# Adoption strategy

MetaFieldRanker is a Haystack 2.0 component. As it is not a breaking change, it should be easy to adopt in combination with the other components.

# How we teach this

It would be sufficient to integrate a small comparison with the RecentnessRanker in the documentation.


# Unresolved questions

The main issue is the implementation strategy. Especially if we want to include the `ranking_mode` parameter in the
MetaFieldRanker, it would make sense to have the RecentnessRanker as a subclass of the MetaFieldRanker.
