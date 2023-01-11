- Title: Shaper
- Decision driver: Vladimir
- Start Date: 2022-12-29
- Proposal PR: https://github.com/deepset-ai/haystack/pull/3784/

# Summary

Input/Output Shaper (Shaper) is a new pipeline component that can invoke arbitrary, registered functions, on the
invocation context (query, documents etc.) of a pipeline and pass the new/modified variables further down the pipeline.

# Basic example

In the following example, we'll use Shaper to add a new variable `questions` to the invocation context.
`questions` is a copy of query variable. This functionality of Shaper is useful when we simply want to
rename a variable in the invocation context e.g. in cases where the PromptNode template is expecting a variable
'questions' rather than 'query'.


```python

    from haystack import Pipeline, Document

    with open("tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              params:
                inputs:
                    query:
                      output: questions
              type: Shaper
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path="tmp_config.yml")
    result = pipeline.run(
        query="What can you tell me about Berlin?",
        documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
    )
    assert result
    # query has been renamed to questions
    assert isinstance(result["meta"]["invocation_context"]["questions"], str)


```

# Motivation

We need Shaper to support the use cases where we want to easily add new variables to the pipeline invocation context.
These new variables hold values which are a result of some arbitrary function invocation on the existing variables
in the invocation context.

Shaper is especially useful when combined with PromptNode(s). Aside from simply renaming variables to match
the templates of PromptNodes, we can also use Shaper to add new variables to the invocation context. Often
these new variables are the result of some arbitrary function invocation on the existing variables in the
invocation context.

The original idea for Shaper is related to question answering use case using PromptNode. In QA, query string variable
passed to a pipeline run method needs to be expanded to a list of strings with the list size matching the size of the
documents list. Therefore, we can use the query as the question to pose to all the documents in the documents list.

The expected outcome of using Shaper is that we can easily add new variables to the invocation context so they can
match the prompt templates of PromptNodes. Multiple Shaper components can be used in a pipeline to modify the
invocation context as needed.


# Detailed design

The Shaper component is most often defined in pipelines YAML file. The YAML component definition consists of the
params block:

```yaml
            components:
            - name: shaper
              params:
                inputs:
                   query:
                      func: expand
                      output: questions
                      params:
                        expand_target: query
                        size:
                          func: len
                          params:
                            - documents
                   documents:
                      func: concat
                      params:
                        docs: documents
                        delimiter: " "
              type: Shaper
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
```

The params block consists of the inputs list. Each item in the inputs list is a dictionary with the key being the
invocation context variable that we want to modify.

In the example above, we have two items in the inputs list. The first item is a dictionary with the key `query` and the
second item is a dictionary with the key `documents`.

For the query variable, we want to invoke the function `expand` and store the result in the variable `questions`.
The `expand` function takes two keyword parameters: `expand_target` and `size`. The `expand_target` parameter is the
name of the variable in the invocation context that we want to expand. The `size`parameter is a result of the `len`
function invocation on the variable `documents`.

For the documents variable, we want to invoke the function `concat` and store the result in the same variable.
Therefore, after the invocation, the documents variable will hold a result of `concat` function invocation while
we'll also have a new variable `questions` in the invocation context. The questions variable will hold a result of
`expand` function invocation.

The important thing to note here is that we can invoke functions with both keyword and positional parameters. Function
`len` is an example of a function that takes non-keyword positional parameters. The `concat` and `expand` function
take keyword parameters. These functions can also be invoked with positional parameters but that is not recommended.


### Default parameters

The Shaper component can also be configured with default parameters. Default parameters are used when we
don't specify the parameters for a function invocation. The default parameters are specified in the function definition.

For example, in the YAML snippet definition below, we have a function `expand` that takes two keyword parameters:
`expand_target` and `size`. However, we haven't specified either of these parameters in the YAML config. This is
possible because we assume that the first parameter is always the variable we want to invoke the function on. In this
case, the variable `query`. The second parameter is the `size` of the list we want to expand the variable to. Here we
have also defined a helper function in Shaper called `expand:size` that calculates the default value of
this parameter - `len(documents)`.

Therefore, the `expand` function, described below, will be invoked with the following parameters: `query`
and `len(documents)`

```yaml
            components:
            - name: shaper
              params:
                inputs:
                    query:
                      func: expand
                      output: questions
              type: Shaper
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query

```

We need the "default parameter" functionality to support YAML files definitions that are more concise and less
error-prone.


### Omitting output parameter

The output parameter is optional. If it is omitted, the result of the function invocation will be stored in
the corresponding input variable. In the example below, the output of expand function will be stored in the
query variable.

```yaml
            components:
            - name: shaper
              params:
                inputs:
                    query:
                      func: expand
              type: Shaper
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query

```

### Order of function invocation

The order of function invocation is important. The functions are invoked in the order they are defined in the YAML.
In the example below, we have two input variables: `query` and `documents`. The `query` variable is expanded to a
list of strings and stored in the variable `questions`. The `documents` variable is then contracted and the
variable `questions` is immediately used as the `num_tokens` keyword parameter to the `concat` function.

```yaml
            components:
            - name: shaper
              params:
                inputs:
                    query:
                      func: expand
                      output: questions
                      params:
                        expand_target: query
                        size:
                          func: len
                          params:
                            - documents
                    documents:
                      func: concat
                      output: documents
                      params:
                        docs: documents
                        delimiter: " "
                        num_tokens:
                            func: len
                            params:
                                - questions
              type: Shaper
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
```

### Security

The Shaper component prevents arbitrary code execution. There should be no dangerous `exec` or `eval` Python calls. Only
the functions we have specified in the Shaper component are allowed to be invoked. The functions are specified in the
constructor using immutable data structures.

This security measure might be relaxed in the future to allow users to specify their own functions. However, this
change will require a more thorough security review.

# Drawbacks and other considerations

Although a "normal use" of PromptNodes would not trigger a need for Shaper there are cases where its
use is necessary. In cases where we can only use pipeline definitions to configure the pipeline (via YAML),
we need to use it.


- Implementation and maintenance cost should not be high.
- Shaper is not really useful outside of Haystack pipeline.
- Shaper could turn out to be useful in other use cases as well - i.e. declarative pre/post processing.
- Shaper integrates well with PromptNodes and other components.
- No braking changes to existing components.


# Alternatives

A better solution would likely be a more general run method for components. This would allow us to arbitrarily
define the pipeline invocation context. However even in those cases we'll need to use Shaper to modify
existing variables in invocation context as needed.

# Adoption strategy

Haystack users can start using Shaper in their pipelines immediately. There are no breaking changes to
existing components or pipelines.


# How we teach this

We will need docs update to teach users how to use Shaper. The docs will need to explain the
motivation using Shaper and PromptNode examples. We also need to show the usage via tutorials.


# Unresolved questions

Optional, but suggested for first drafts. What parts of the design are still
TBD?
