- Title: Agent tools
- Decision driver: Vladimir Blagojevic (with Julian Risch)
- Start Date: 2023-02-06
- Proposal PR: https://github.com/deepset-ai/haystack/pull/4084


# Summary

Agent system consists of an extendable set of modules (aka experts/tools) and a "router" dispatching natural
language input to a module that can best respond to the given input. These modules can be:

• Neural, including the general-purpose LLMs as well as other smaller, specialized LMs.

• Symbolic, for example a math calculator, a currency converter, or an API call to a search engine.


Current Haystack pipeline design approach fits nicely with these extension modules/tools. They can be tested in
isolation and subsequently integrated into the agent. The agent can register these tools, understand their objectives
and route the user input to the appropriate tool. The tools can be either Haystack pipelines or components.


A demo idea for the new Agent is to demonstrate Question Answering on Technical Documentation (using the example of Haystack). It should be useful and impressive, and it determines what tools we will implement first.

**Example Questions:**
- "Why am I seeing duplicate answers being returned?" based on indexed FAQ documentation
- "Which organizations use Haystack?" based on web search and Wikipedia returning answers via SerpAPI
- "How can I choose the model for PromptNode?" based on retrieving documents via WebRetriever
- "How can I make overwrite_with_env_variables work in RayPipeline" based on an open issue found with GitHub API or web search

If an answer cannot be found in indexed files, the Agent will use self reflection to rephrase the question and/or search the web. It will give updates while searching, for example print thoughts: “Found nothing in indexed documentation. Will continue with web search.” If still nothing can be found, the Agent will generate a link to a pre-filled and pre-tagged issue template like this that the user can choose to create. Tools required for the demo: SerpAPI, GitHubAPI, Self Reflection Module, WebRetriever.

We also need to demonstrate how the Agent uses a combination of multiple tools to answer a question instead of just trying them sequentially.

**Example Question:**
- "Is there an open issue about any of the nodes in a standard QA pipeline not working?"

Here the LLM might first use documentation to find out more about what nodes are part of standard QA pipelines and then searches on GitHub for open issues that mention retriever or reader.

We propose the following tools to be developed for the MVP version of our agent (#3925):

- SerpAPI
- Self reflection module (query rephrasing)
- WebRetriever
- deepset Cloud API

# Basic example

No code examples are provided, but a high-level overview of how the feature would be used.

There is also a [colab notebook](https://colab.research.google.com/drive/1oJf4gxkokIazLN26NNt6XsewMvCmVrz8?usp=sharing) demonstrating how to use an agent with tools and a [branch](https://github.com/deepset-ai/haystack/compare/main...mrkl-pipeline) for demo purposes (no pull request).

# Motivation

Agent tools are the main and essential building block of the agent system. They are the modules extending the
agent's capabilities. Agent, when it relies only on itself, is not as powerful as when it can leverage the
external modules (tools) that are highly specialized in their respective domains. Agent can't do, for example, lookup of
restaurants in our neighbourhood, but it can use SerpAPI to do that. When it comes to complex math calculations, LLM
would be a poor choice, but a math calculator would be a good fit, and so on.

Combining the power of LLMs with the power of external tools is the key ingredient to the success of the agent framework.


# Detailed design

In the following sections, we list essential agent tools required for agent MVP. We start with the general design
principles and then describe each tool in detail.

Each tool is defined as a Python class that inherits from the BaseComponent class. The base class acts as a wrapper
around the actual tool implementation.

The main Agent modules/tools are:

## SerpAPI

SerpAPI (Search Engine Results Page) is a symbolic API module allowing programmatic interaction with Google and other
search engines. We can use Serp directly via REST API or the Python library provided by SerpAPI.

Although https://serpapi.com/ provides a Python library, we will implement our Python wrapper around the REST API.
The wrapper will allow us to use the same SerpAPI module for other search engines (e.g. Bing, DuckDuckGo, etc.) that
SerpAPI already supports.

We'll wrap the SerpAPI module with our best practices regarding retrying failed requests, handling timeouts, etc.
We'll also enable our standard approach of injecting the API key via environment variables.

### SerpAPI scoping

A great feature of SerpAPI is that it can be scoped to a particular domain. Therefore, in our demo we can search through
Haystack documentation on docs.haystack.com, github.com/deepset-ai/haystack and so on.

## Self reflection module

SRM is a neural module that aims to improve the agent's overall robustness. Agents can sometimes be very fragile
in their execution steps. As a core component, the motivation for the self-reflection module (SRM) is to improve the
robustness of the agent's execution. Agents can be fragile in their execution steps due to the non-deterministic nature
of LLM inferencing and their sensitivity to the chosen prompts.

SRM relies on instruction following LLM and operates in two lock-step sequences. Given any other agent module/tool,
its description, and its input and output - SRM does the following:

1. It checks if the module description (objective), the input, and the output are aligned/congruent.
2. If they are not aligned, it rephrases the input (while retaining semantic meaning) and attempts to elicit an aligned
output.

SRM can improve the robustness of any module. For example, SerpAPI can sometimes be very sensitive to query wording.
If we are searching for "Olivia Wilde's boyfriend", the results might not be as precise as if we search for "Who is
Olivia Wilde's current boyfriend?". SRM can rephrase the query to the latter form if the former form doesn't return
an aligned answer.

It is still an open question of how to implement SRM. We can make it an internal component or another pipeline. The
former approach is more efficient and has tight coupling, and the latter is more flexible. We'll carefully weigh
the pros and cons of each approach and decide on the best approach.

One idea how to integrate SRM into an existing Agent and its tools is that every tool can set a parameter indicating
how often SRM should run for this tool (default value 0). If it is 0, SRM will not be used at all for that tool,
otherwise it will check at least once for alignment/congruency and optionally rephrase. This will repeat but not
more often than the parameter allows.


## WebRetriever

WebRetriever is a symbolic module that allows users to query the web for relevant documents. It is a wrapper around
SerpAPI that produces a list of Haystack Documents.

Given a user query passed via the run method, SerpAPI first fetches the top_k relevant URL hits, which are downloaded
and processed. The processing involves stripping irrelevant HTML tags and producing clean raw text. WebRetriever
then splits raw text into paragraph-long Documents of the desired size. WebRetriever can use the optional
Cross-Encoder to filter out query-irrelevant documents.

In the future, we'll develop WebRetriever variants with DocumentStore that caches documents with some expiration
setting. The enhanced WebRetreiever versions will allow us to avoid downloading the same documents from the web
multiple times.

However, for the first version of the agent, we will keep WebRetriever as simple as possible.

## deepset Cloud API
The Agent should be able to use pipelines deployed on deepset Cloud as a tool.
To this end, the text question needs to be send via REST API to the [search endpoint](https://docs.cloud.deepset.ai/reference/search_api_v1_workspaces__workspace_name__pipelines__pipeline_name__search_post) of a given pipeline deployed on deepset Cloud.

In the demo, the Agent will use this tool for question answering on indexed documents of the Haystack documentation, such as documentation web pages or tutorials.


# Drawbacks

One of the main reasons why we should not work on implementing this proposal is that it is a rather large
undertaking requiring substantial resources. This naturally carries significant risks. We should carefully weigh
the pros and cons of tools we want to implement and prioritize them. The current priority criteria is that the tool
should be useful for the agents MVP, the majority of early adopters and that it should be relatively easy to
implement. We can always add more tools in the future.

Another priority guidance is the demo we intend to build. We want to build a demo that showcases the agent's
capabilities in a use case that is relevant to the majority of early adopters (current Haystack users).


# Alternatives

We also considered the following demo alternatives:

- Medical QA: We can build a demo that answers medical questions. This is a very interesting use case but also
fraught with risks. A question one might ask in such a demo is “Which antibiotic should I use for urinary tract infections?”.
A factoid-based QA system might (reasonably) return the answer “trimethoprim 200mg”. However, a “correct” answer is not
sufficient to translate into clinical use. There were other recent demos but they were not very successful.
See https://twitter.com/GlassHealthHQ/status/1620092094034620421 for more details.

- Public Healthcare QA: a bit less risky proposal than the medical QA. We can build a demo that answers questions about
healthy diet, cooking recipes, vitamines etc. This demo would use almost exactly the same tools as the main demo proposal
and we can potentially switch to this demo if needed.

- Financial Domain (earnings transcript): we can build a demo that answers questions about earnings transcripts. However,
we were not sure if this is a good use case for the agent as it is not very relevant to the majority of early adopters.


# Adoption strategy

An adoption strategy for this proposal is not needed as much and the objective is to demo the capabilities of the agent
and inspire early adopters to use the agent and the main tools we are going to implement.


# Demo

See the Summary section for the demo description.

# How we teach this

We intend to use the demo to teach users about the agent's capabilities. We'll subsequently add more documentation about
core components used in the demo and the agent in general. This demo would be mainly used to promote Haystack Agents and
to generate interest in the agent.

# Unresolved questions

Optional, but suggested for first drafts. What parts of the design are still
TBD?
