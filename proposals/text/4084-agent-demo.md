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


A demo idea for the new Agent is to demonstrate Question Answering on Technical Documentation (using the example of
Haystack). It should be useful and impressive, and it determines what tools we will implement first.

**Example Questions:**
- "Why am I seeing duplicate answers being returned?" based on indexed FAQ documentation
- "Which organizations use Haystack?" based on web search and Wikipedia returning answers via SearchEngine
- "How can I choose the model for PromptNode?" based on retrieving documents via WebRetriever
- "How can I make overwrite_with_env_variables work in RayPipeline" based on an open issue found with GitHub API or web search

If an answer cannot be found in indexed files, the Agent will use self reflection to rephrase the question and/or search the
web. It will give updates while searching, for example print thoughts: “Found nothing in indexed documentation. Will continue
with web search.” If still nothing can be found, the Agent will generate a link to a pre-filled and pre-tagged issue template
like this that the user can choose to create. Tools required for the demo: SearchEngine, GitHubAPI, Self Reflection Module, WebRetriever.

We also need to demonstrate how the Agent uses a combination of multiple tools to answer a question instead of just trying
them sequentially.

**Example Question:**
- "Is there an open issue about any of the nodes in a standard QA pipeline not working?"

Here the LLM might first use documentation to find out more about what nodes are part of standard QA pipelines and then
searches on GitHub for open issues that mention retriever or reader.

We propose the following tools to be developed for the MVP version of our agent (#3925):

- SearchEngine
- Self reflection module (query rephrasing)
- WebRetriever
- Top-p (nucleus) sampling
- Agent memory
- deepset Cloud API

# Basic example

No code examples are provided, but a high-level overview of how the feature would be used.

There is also a [colab notebook](https://colab.research.google.com/drive/1oJf4gxkokIazLN26NNt6XsewMvCmVrz8?usp=sharing)
demonstrating how to use an agent with tools and a [branch](https://github.com/deepset-ai/haystack/compare/main...mrkl-pipeline)
for demo purposes (no pull request).

# Motivation

Agent tools are the main and essential building block of the agent system. They are the modules extending the
agent's capabilities. Agent, when it relies only on itself, is not as powerful as when it can leverage the
external modules (tools) that are highly specialized in their respective domains. Agent can't do, for example, lookup of
restaurants in our neighbourhood, but it can use SearchEngine to do that. When it comes to complex math calculations, LLM
would be a poor choice, but a math calculator would be a good fit, and so on.

Combining the power of LLMs with the power of external tools is the key ingredient to the success of the agent framework.


# Detailed design

In the following sections, we list essential agent tools required for agent MVP. We start with the general design
principles and then describe each tool in detail.

Each tool is defined as a Python class that inherits from the BaseComponent class. The base class acts as a wrapper
around the actual tool implementation.

The main Agent modules/tools are:

## SearchEngine

SearchEngine is a symbolic API module allowing programmatic interaction with Google and other search engines. We'll have
multiple providers of SearchEngine including https://serper.dev and https://serpapi.com as initial providers.

SearchEngine will return a list of results (e.g. List[Document]), the content of each document being a "snippet" of the
single search result, while all other attributes of the search results (e.g. title, url link, etc.) will
be metadata of the document.

### SearchEngine scoping

A great feature of SearchEngine is that it can be scoped to a particular domain. Therefore, in our demo, if so desired,
we can search through Haystack documentation on docs.haystack.com, github.com/deepset-ai/haystack and so on.

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

SRM can improve the robustness of any module. For example, SearchEngine can sometimes be very sensitive to query wording.
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

One possible implementation of SRM could be https://arxiv.org/abs/2203.11171

The main motivation for addition of SRM (and self-consistency ideas in general) is the finding from
https://arxiv.org/abs/2210.03629 Google Brain researchers discovered that "Wrong reasoning trace (including failing
to recover from repetitive steps)" accounts for 47% of agent errors.

## WebRetriever

WebRetriever is a symbolic module that allows users to query the web for relevant documents. It is a wrapper around
SearchEngine that produces a list of Haystack Documents.

WebRetriever will operate in two modes:

- snippet mode: WebRetriever will return a list of Documents, each Document being a snippet of the search result
- document mode: WebRetriever will return a list of Documents, each Document being a full HTML stripped document of the search result

In document mode, given a user query passed via the run method, SearchEngine first fetches the top_k relevant URL hits, which are
downloaded and processed. The processing involves stripping irrelevant HTML tags and producing clean raw text. WebRetriever
then splits raw text into paragraph-long Documents of the desired size.

In the future, we'll develop WebRetriever variants with DocumentStore that caches documents with some expiration
setting. The enhanced WebRetreiever versions will allow us to avoid downloading the same documents from the web
multiple times.

However, for the first version of the agent, we'll strive to keep WebRetriever as simple as possible.

## Top-p (nucleus) sampling

Although very useful, top-k ranking is sometimes inferior to top-p ranking. Instead of filtering only from the most
likely k hits, in top-p sampling we choose the smallest possible set of documents whose cumulative probability of
relevance exceeds the probability p (usually close to 1). The relevance could be calculated via sbert.net CrossEncoder using
query and the document content.

In web search, this is a very useful feature as it allows us to avoid query irrelevant documents and be super precise
in our search results. We'll implement top-p sampling as a separate module that WebRetriever can use. Other components
in Haystack can use it as well.

The main motivation for addition of top-p sampling is the finding from https://arxiv.org/abs/2210.03629 Google Brain
researchers found that "Search result error" is the main cause in 23% of the cases of agent failure. Top-p sampling
can help us minimize this point of failure.

Note that one can still use top-k filtering via Ranker and top-p filtering via TopPSampler in combination.

## Agent memory

Although we currently support only so-called ReAct agents, it is not hard to envision a future where we'll have
additional agent types including conversational agents.

Due to LLMs one-shot-forget nature of inferencing, conversational agents might need to remember the context of the
conversation. To support conversational agents, we'll need Agent memory component. The memory component will initially contain
two submodules: entity extraction and summarization

Entity extraction is a neural module extracting entities from the provided conversation transcript (raw text).
The entities are best thought of as an outcome of Named-entity recognition task; for example, people, places, organizations etc.

Entity summarization is a neural module that summarizes the entities extracted by the entity extraction module.

Entity extraction and summarization are run in the background as the conversation progresses. The
frequency of extraction and summarization updates will be configurable.

The extracted entities along with relevant summaries will be stored in the Agent memory. Agent memory implementation
details are out of scope of this proposal; they could be various short or long term memory storage options.

For the first version of the agent, we'll strive to keep the memory component as simple as possible; we'll
only implement entity extraction and summarization while we'll use runtime memory for storing entities.

### Future improvements:

As we have limited token payload for model inferencing, we'll need to implement a mechanism for decaying memory.

If we have many entities in the memory, we'll also need to implement a mechanism for entities selection.
We'll likely need no summaries for well-known entities like "Elon Musk" or "New York".

Information related to the entities could become stale over time (e.g."I'm currently in New York") and we'll
need to implement a mechanism for updating the entities.

The mechanism for decaying memory, prioritizing and updating entities is out of scope of this proposal.


## deepset Cloud API
The Agent should be able to use pipelines deployed on deepset Cloud as a tool.
To this end, the text question needs to be send via REST API to
the [search endpoint](https://docs.cloud.deepset.ai/reference/search_api_v1_workspaces__workspace_name__pipelines__pipeline_name__search_post) of a given pipeline deployed on deepset Cloud.

In the demo, the Agent will use this tool for question answering on indexed documents of the Haystack documentation,
such as documentation web pages or tutorials.


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
