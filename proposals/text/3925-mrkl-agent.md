- Title: MRKLAgent
- Decision driver: @julian-risch (in close collaboration with @vblagoje )
- Start Date: 2023-01-24
- Proposal PR: https://github.com/deepset-ai/haystack/pull/3925
- Github Issue or Discussion: https://github.com/deepset-ai/haystack/issues/3753

# Summary
The MRKLAgent class answers queries by choosing between different actions/tools, which are implemented as pipelines.
It uses a large language model (LLM) to generate a thought based on the query, choose an action/tool, and generate the input for the action/tool.
Based on the result returned by an action/tool, the MRKLAgent has two options.
It can either repeat the process of 1) thought, 2) action choice, 3) action input or it stops if it knows the answer.

The MRKLAgent can be used for questions that contain multiple subquestions that can be answered step-by-step (Multihop QA).
Combined with tools like Haystack's PythonRuntime or SerpAPIComponent, the MRKLAgent can query the web and do calculations.

We have a notebook that demonstrates how to use MRKLAgent. It requires API keys for OpenAI and SerpAPI: https://colab.research.google.com/drive/1oJf4gxkokIazLN26NNt6XsewMvCmVrz8?usp=sharing
The notebook is based on the branch https://github.com/deepset-ai/haystack/compare/main...mrkl-pipeline (no pull request)

# Basic example

An example of a MRKLAgent could use two tools: a web search engine and a calculator.

For example, the query "Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?" can be broken down into three steps:
1) Searching the web for the name of Olivia Wilde's boyfriend
2) Searching the web for the age of that boyfriend
3) Calculating that age raised to the 0.23 power

And the MRKLAgent would respond in the end with "Jason Sudeikis, Olivia Wilde's boyfriend, is 47 years old and his age raised to the 0.23 power is 2.4242784855673896." A detailed walk-through follows below.


# Motivation

With MRKLAgent, users can combine multiple LLMs and tools, so that they can build a truly powerful app. They can use an LLM in a loop to answer more complex questions than with ExtractiveQA or GenerativeQA available in Haystack. With MRKLAgent and tools for web search, Haystack is not limited to extracting answers from a document store or generating answers based on model weights only but it can use the knowledge it retrieves on-the-fly from the web.

In future, we envision that MRKLAgent could use tools not only for retrieving knowledge but also for interacting with the world. For example, it could periodically skim through newly opened issues on GitHub for questions that can be answered based on documentation. It could then retrieve relevant pages from the documentation, generate an answer and post it as a response to the issue.

# Detailed design

This is the bulk of the proposal. Explain the design in enough detail for somebody familiar with Haystack to understand, and for somebody familiar with the implementation to implement. Get into specifics and corner-cases, and include examples of how the feature is used. Also, if there's any new terminology involved, define it here.

The MRKLAgent consists of a PromptNode that generates thoughts, chooses actions, and generates action inputs.
It has an attribute that contains a description of the tools it can choose from. Another attribute contains information on how to invoke each tool by mapping each tool name to a pipeline.

Just like Haystack pipelines, a MRKLAgent can be loaded from a YAML file. That YAML file must also contain the tools of the MRKLAgent defined as pipelines.
A key functionality of the MRKLAgent is that it can act iteratively and execute any of the pipelines as many times as it wants based on the input query and the results returned from the pipelines. In every iteration, it chooses one of the tool pipelines and generates the input to that pipeline dynamically.

An example application of this is MultiHopQA, where multiple subquestions need to be answered step-by-step.
For the example query "Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?", the MRKLPipeline needs to answer several subquestions. Here is an example of a full transcript of the prompt input and generated output:

```
Answer the following questions as best as you can. You have access to the following tools:

Search: useful for when you need to answer questions about current events. You should ask targeted questions
Calculator: useful for when you need to answer questions about math

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Search, Calculator]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
Question: Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?
Thought: I need to do some research to answer this question.
Action: Search
Action Input: Olivia Wilde's boyfriend
Observation: First linked in November 2011, Wilde and Sudeikis got engaged in January 2013. They later became parents, welcoming son Otis in 2014 and daughter Daisy in 2016.
Thought: I need to find out his age
Action: Search
Action Input: Jason Sudeikis age
Observation: 47 years
Thought: I need to raise it to the 0.23 power
Action: Calculator
Action Input: 47^0.23
Observation: 2.4242784855673896
Thought: I now know the final answer
Final Answer: Jason Sudeikis, Olivia Wilde's boyfriend, is 47 years old and his age raised to the 0.23 power is 2.4242784855673896.
```

The MRKLAgent can be either created programmatically or loaded from a YAML file:

**Example programmatic creation:**
```yaml
    prompt_model = PromptModel(model_name_or_path="text-davinci-003", api_key=os.environ.get("OPENAI_API_KEY"))
    prompt_node = PromptNode(model_name_or_path=prompt_model, stop_words=["Observation:"])
    search = SerpAPIComponent(api_key=os.environ.get("SERPAPI_API_KEY"))
    search_pipeline = Pipeline()
    search_pipeline.add_node(component=search, name="Serp", inputs=["Query"])

    tools = [
        {
            "pipeline_name": "serpapi_pipeline",
            "tool_name": "Search",
            "description": "useful for when you need to answer questions about current events. You should ask targeted questions",
        }
    ]
    tool_map = {"Search": search_pipeline}
    mrkl_agent = MRKLAgent(prompt_node=prompt_node, tools=tools, tool_map=tool_map)
    result = mrkl_agent.run(query="What is 2 to the power of 3?")
```

**Example YAML file:**
```yaml
    version: ignore
    components:
      - name: MRKLAgent
        type: MRKLAgent
        params:
          prompt_node: MRKLAgentPromptNode
          tools: [{'pipeline_name': 'serpapi_pipeline', 'tool_name': 'Search', 'description': 'useful for when you need to answer questions about current events. You should ask targeted questions'}, {'pipeline_name': 'calculator_pipeline', 'tool_name': 'Calculator', 'description': 'useful for when you need to answer questions about math'}]
      - name: MRKLAgentPromptNode
        type: PromptNode
        params:
          model_name_or_path: DavinciModel
          stop_words: ['Observation:']
      - name: DavinciModel
        type: PromptModel
        params:
          model_name_or_path: 'text-davinci-003'
          api_key: 'XYZ'
      - name: Serp
        type: SerpAPIComponent
        params:
          api_key: 'XYZ'
      - name: CalculatorInput
        type: PromptNode
        params:
          model_name_or_path: DavinciModel
          default_prompt_template: CalculatorTemplate
          output_variable: python_runtime_input
      - name: Calculator
        type: PythonRuntime
      - name: CalculatorTemplate
        type: PromptTemplate
        params:
          name: calculator
          prompt_text:  |
              # Write a simple python function that calculates
              # $query
              # Do not print the result; invoke the function and assign the result to final_result variable
              # Start with import statement
    pipelines:
      - name: mrkl_query_pipeline
        nodes:
          - name: MRKLAgent
            inputs: [Query]
      - name: serpapi_pipeline
        nodes:
          - name: Serp
            inputs: [Query]
      - name: calculator_pipeline
        nodes:
          - name: CalculatorInput
            inputs: [Query]
          - name: Calculator
            inputs: [CalculatorInput]
"""
```

# Drawbacks

Look at the feature from the other side: what are the reasons why we should _not_ work on it? Consider the following:

- What's the implementation cost, both in terms of code size and complexity?
- Can the solution you're proposing be implemented as a separate package, outside of Haystack?
- Does it teach people more about Haystack?
- How does this feature integrate with other existing and planned features?
- What's the cost of migrating existing Haystack pipelines (is it a breaking change?)?

There are tradeoffs to choosing any path. Attempt to identify them here.

# Alternatives

What other designs have you considered? What's the impact of not adding this feature?

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

**Additional MRKLPipeline**
In the current proposal we load a MRKLAgent from YAML although it is not a Pipeline but a BaseComponent.
An alternative is to create also a MRKLPipeline that is loaded from YAML instead and which contains a MRKLAgent.
That way we better follow the existing concept of pipelines in Haystack. It would result in a pipeline consisting of other pipelines.

**Name of MRKLAgent**
- MRKLAgent
- LLMOrchestrator
- LLMChain
- PipelineComposer / LLMComposer
- PipelineComposition / LLMComposition
- Interesting naming tidbits:
- MRKL [paper](https://arxiv.org/pdf/2205.00445.pdf) never uses word agent, only system
- ReAct [paper](https://arxiv.org/pdf/2210.03629.pdf) uses agent almost exclusively

**Tools we imagine in the near future**
- PlotGenerator / Plotter
