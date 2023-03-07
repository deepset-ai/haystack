- Title: Shapers in Prompt Templates
- Decision driver: tstadel
- Start Date: 2023-02-15
- Proposal PR: (fill in after opening the PR)
- Github Issues or Discussion:
  - spike: https://github.com/deepset-ai/haystack/pull/4061
  - solved issues:
    - https://github.com/deepset-ai/haystack/issues/3877
    - https://github.com/deepset-ai/haystack/issues/4053
    - https://github.com/deepset-ai/haystack/issues/4047

# Summary

In order to make prompt templates more flexible and powerful while at the same time making PromptNode as easy to use as any other node in Haystack, we want to introduce two modifications to PromptTemplate:
- output: support Shapers in PromptTemplates to enable the user to define how the output to the prompt template should be shaped
- input: extend the prompt syntax to support the usage of functions that can be applied to input variables

With these modifications prompt templates will be able to define, and abstract away from PromptNode, everything that is necessary to create a Haystack node that is specialized for a certain use-case (e.g. generative QA). Additionally, PromptTemplates will be fully serializable, enabling everyone to share their prompt templates with the community.

# Basic example

A generative QA pipeline would be as easy as this:

    ```python
    from haystack import Pipeline
    from haystack.document_store import InMemoryDocumentStore
    from haystack.nodes import PromptNode, EmbeddingRetriever

    document_store = InMemoryDocumentStore()
    retriever = EmbeddingRetriever(document_store=document_store, ...)
    pn = PromptNode(default_prompt_template="question-answering-with-references")

    p = Pipeline()
    p.add_node(component=retriever, name="Retriever", inputs=["Query"])
    p.add_node(component=pn, name="Prompt", inputs=["Retriever"])
    ```

As a result we get a pipeline that uses PromptNode as a drop-in replacement for Generators:

    ```python
    p.run(
        query="What is the most popular drink?"
    )
    ```

    ```python
    {'answers': [<Answer {'answer': 'Potable water is the most popular drink, followed by tea and beer as stated in Document[5].', 'type': 'generative', 'score': None, 'context': None, 'offsets_in_document': None, 'offsets_in_context': None, 'document_ids': ['fcd62336fb380a69c2d655f8cd072995'], 'meta': {}}>],
    'invocation_context': {'query': 'What is the most popular drink?',
    'documents': [<Document: {'content': 'Beer is the oldest[1][2][3] and most widely consumed[4] type of alcoholic drink in the world, and the third most popular drink overall after potable water and tea.[5] It is produced by the brewing and fermentation of starches, mainly derived from cereal grains—most commonly from malted barley, though wheat, maize (corn), rice, and oats are also used. During the brewing process, fermentation of the starch sugars in the wort produces ethanol and carbonation in the resulting beer.[6] Most modern beer is brewed with hops, which add bitterness and other flavours and act as a natural preservative and stabilizing agent. Other flavouring agents such as gruit, herbs, or fruits may be included or used instead of hops. In commercial brewing, the natural carbonation effect is often removed during processing and replaced with forced carbonation.[7]', 'content_type': 'text', 'score': None, 'meta': {}, 'id_hash_keys': ['content'], 'embedding': None, 'id': 'fcd62336fb380a69c2d655f8cd072995'}>],
    'answers': [<Answer {'answer': 'Potable water is the most popular drink, followed by tea and beer as stated in Document[5].', 'type': 'generative', 'score': None, 'context': None, 'offsets_in_document': None, 'offsets_in_context': None, 'document_ids': ['fcd62336fb380a69c2d655f8cd072995'], 'meta': {}}>]},
     '_debug': {'PromptNode': {'runtime': {'prompts_used': ['Create a concise and informative answer (no more than 50 words) for a given question based solely on the given documents. You must only use information from the given documents. Use an unbiased and journalistic tone. Do not repeat text. Cite the documents using Document[number] notation. If multiple documents contain the answer, cite those documents like ‘as stated in Document[number,number,etc]’. If the documents do not contain the answer to the question, say that ‘answering is not possible given the available information.’\nDocument[1]: Beer is the oldest(1)(2)(3) and most widely consumed(4) type of alcoholic drink in the world, and the third most popular drink overall after potable water and tea.(5) It is produced by the brewing and fermentation of starches, mainly derived from cereal grains—most commonly from malted barley, though wheat, maize (corn), rice, and oats are also used. During the brewing process, fermentation of the starch sugars in the wort produces ethanol and carbonation in the resulting beer.(6) Most modern beer is brewed with hops, which add bitterness and other flavours and act as a natural preservative and stabilizing agent. Other flavouring agents such as gruit, herbs, or fruits may be included or used instead of hops. In commercial brewing, the natural carbonation effect is often removed during processing and replaced with forced carbonation.(7); \n Question: What is the most popular drink?; Answer: ']}}},
     'root_node': 'Query',
     'params': {},
     'query': 'What is the most popular drink?',
     'documents': [<Document: {'content': 'Beer is the oldest[1][2][3] and most widely consumed[4] type of alcoholic drink in the world, and the third most popular drink overall after potable water and tea.[5] It is produced by the brewing and fermentation of starches, mainly derived from cereal grains—most commonly from malted barley, though wheat, maize (corn), rice, and oats are also used. During the brewing process, fermentation of the starch sugars in the wort produces ethanol and carbonation in the resulting beer.[6] Most modern beer is brewed with hops, which add bitterness and other flavours and act as a natural preservative and stabilizing agent. Other flavouring agents such as gruit, herbs, or fruits may be included or used instead of hops. In commercial brewing, the natural carbonation effect is often removed during processing and replaced with forced carbonation.[7]', 'content_type': 'text', 'score': None, 'meta': {}, 'id_hash_keys': ['content'], 'embedding': None, 'id': 'fcd62336fb380a69c2d655f8cd072995'}>],
     'node_id': 'PromptNode'}
    ```

The corresponding prompt template would look like this (provided `join_documents` and `strings_to_answers` Shaper functions are extended a bit):

    ```python
    PromptTemplate(
            name="question-answering-with-references",
            prompt_text="Create a concise and informative answer (no more than 50 words) for a given question "
            "based solely on the given documents. You must only use information from the given documents. "
            "Use an unbiased and journalistic tone. Do not repeat text. Cite the documents using Document[number] notation. "
            "If multiple documents contain the answer, cite those documents like ‘as stated in Document[number,number,etc]’. "
            "If the documents do not contain the answer to the question, say that ‘answering is not possible given the available information.’\n"
            "{join(documents, '\n', '\nDocument[$idx]: $content', {'\n': ' ', '[': '(', ']': ')'})} \n Question: {query}; Answer: ",
            output_shapers=[
                Shaper(
                    func="strings_to_answers",
                    inputs={"strings": "results", "documents": "documents"},
                    outputs=["answers"],
                )
            ],
            output_variable="answers",
        )
    ```

We make sure that we have proper default values for the input shaping function and it is easy to understand. `{join(documents)}` should be usable in most cases. When you want to have more control over document rendering something like `join(documents, DELIMITER, PATTERN, CHAR_REPLACEMENT)` with

    ```python
    DELIMITER = "\n"
    PATTERN = "$content" # parsable by StringTemplate using data from document.content, document.meta and the index of the document
    CHAR_REPLACEMENT = {"[": "(", "}": ")"} # just an example what could be passed here
    ```

would do.

Note that the number of how many prompts are created depends on which shaping functions are used. If you use `join(documents)` you will have only one prompt. If you omit the `join` and use `to_list(query)` instead, you will have multiple prompts (one prompt per document).

# Motivation

Currently using PromptNode is a bit cumbersome as:
- for using it in popular use-cases like question-answering, it requires to add the Shapers to the pipeline manually which creates a lot of boilerplate code and is not very intuitive
- to customize a prompt within a pipeline, you may need to change four different things: the prompt node, the prompt template, the input shapers and the output shapers. This is not ideal as it requires to write a lot of boilerplate code and makes it hard to iterate quickly on prompts.
- if you wanted to share your prompt template with the community, you would need to share the whole pipeline (as you do need shapers), which is not ideal as it may contain other nodes that are not relevant.


# Detailed design

## General changes
PromptTemplate gets one new attribute: `output_shapers`. These are lists of Shaper objects that are applied to the output of the prompt.
PromptTemplate's syntax is extended to allow for the usage of shaping functions on input variables. These shaping functions are predefined.

## Basic flow:
PromptNode calls `PromptTemplate.prepare` before executing the prompt. `PromptTemplate.prepare` applies the shaping functions (if present) to the arguments of the `invocation_context`.
PromptNode invokes the prompt on the prepared `invocation_context`.
PromptNode calls `PromptTemplate.post_process` after executing the prompt. `PromptTemplate.post_process` makes all `output_shapers` run on the `invocation_context`.

## Shaping functions
The PromptTemplate syntax is extended to allow for the usage of shaping functions on input variables. These shaping functions should be easy to understand and use.
We only support positional args for shaping functions. This is because we want to keep the syntax simple and we don't want to overcomplicate the parsing logic. As args any python primitive is allowed (e.g. strings, ints, floats, lists, dicts, None).
Parsing is done by using regular expressions. If we however notice that this is not enough, we can switch to a more complex parsing library like `jinja2`.
Here is a basic (and incomplete) example how the parsing logic could look like:

    ```python

        # template allowing basic list comprehensions to create the wanted string
        template = """
        Create a concise and informative answer (no more than 50 words) for a given question
        based solely on the given documents. You must only use information from the given documents.
        Use an unbiased and journalistic tone. Do not repeat text. Cite the documents using Document[number] notation.
        If multiple documents contain the answer, cite those documents like ‘as stated in Document[number,number,etc]’.
        If the documents do not contain the answer to the question, say that ‘answering is not possible given the available information.
        {join(documents, new_line)} \n Question: {query}; Answer:
        """

        for group in re.findall(r'\{(.*?)\}', template):
            if "(" in group and ")" in group:
                function_to_call = group[:group.index("(")].strip()
                variables_for_function = [var.strip() for var in group[group.index("(")+1:group.index(")")].split(",")]
                print(f"Found function '{function_to_call}' with vars '{variables_for_function}'")
            else:
                print("Found single variable:", group)

        # Returns
        # >>> Found function 'join' with vars '['documents', 'new_line']'
        # >>> Found single variable: query
    ```

## Prompt engineering with Haystack Pipelines
Additionally we want to support changing the prompt via a param of `Pipeline.run`. This is useful for example if you want to fine-tune your prompt and iterate quickly on it without having to change the pipeline. The `prompt` param is a string in `Pipeline.run` which will be delegated to the `PromptNode` and then used by `PromptTemplate`. This is similar to how `Pipeline.run` works with the `query` param. Note that the `prompt` param does not affect `output_shapers`.

## Misc
Note, that `Shapers` are still usable in Pipelines as before.

# Drawbacks

Look at the feature from the other side: what are the reasons why we should _not_ work on it? Consider the following:

- What's the implementation cost, both in terms of code size and complexity? A good day
- Can the solution you're proposing be implemented as a separate package, outside of Haystack? No
- Does it teach people more about Haystack? No, but it makes it easier to use especially for beginners.
- How does this feature integrate with other existing and planned features? It doesn't change any existing features and should nicely integrate with agents.
- What's the cost of migrating existing Haystack pipelines (is it a breaking change?)? None

It also fosters a bit the nesting of components in Haystack. Although the whole PromptNode ecosystem already does this (e.g. via PromptModel, PromptTemplate being used by PromptNode), it's still a bit of a new concept. However, I think it's a good one and it's not too hard to understand.

We still don't have access to PromptNode, PromptModel or the invocation layer inside of PromptTemplates. If we want PromptTemplate to access fundamental parts of them (e.g. the tokenizer), we would need to pass them to the PromptTemplate. This would make the whole system more complex, but it would be possible.

# Alternatives

Sub-classing specialized PromptNodes like QuestionAnsweringPromptNode, which would have the shapers already defined. This would make it easier to use, but it would be harder to iterate quickly on prompts, be less flexible and sharing is difficult. The same is true for sub-classing PromptTemplate like QuestionAnsweringPromptTemplate. Both sub-classing approaches would make it easier to use, but it would be harder to iterate quickly on prompts, be less flexible and sharing is difficult.

Having `input_shapers` in the same way as `output_shapers` in the PromptTemplate. This would make it harder for users to get started as they would need to understand Shapers and which functions are relevant for input shaping.
# Adoption strategy

As the syntax for input variables in `PromptTemplate` changes we can do the following:
- raise an error if the old syntax is used and tell the user to use the new syntax
- support the old syntax for a while and raise a deprecation warning

# How we teach this

We should show how:
- predefined PromptTemplates can be used
- predefined PromptTemplates can be customized
- custom PromptTemplates can be created

# Unresolved questions

How does `OpenAIAnswerGenerator` make use of input shaping functions and output shapers?
- output shapers: it doesn't use them
- input shaping functions: it uses them if they are present. If not it uses its own default functions.
