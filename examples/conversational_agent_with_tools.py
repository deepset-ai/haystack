import os

from haystack.agents.base import ConversationalAgentWithTools, Tool, ToolsManager
from haystack.agents.memory import ConversationSummaryMemory
from haystack.agents.types import Color
from haystack.nodes import PromptNode, WebRetriever, PromptTemplate
from haystack.pipelines import WebQAPipeline

few_shot_prompt = """
In the following conversation, a human user interacts with an AI Agent using the ChatGPT API. The human user poses questions, and the AI Agent goes through several steps to provide well-informed answers.

If the AI Agent knows the answer, the response begins with "Final Answer:" on a new line.

If the AI Agent is uncertain or concerned that the information may be outdated or inaccurate, it must use the available tools to find the most up-to-date information. The AI has access to these tools:
{tool_names_with_descriptions}

AI Agent responses must start with one of the following:

Thought: [AI Agent's reasoning process]
Tool: [{tool_names}] (on a new line) Tool Input: [input for the selected tool WITHOUT quotation marks and on a new line] (These must always be provided together and on separate lines.)
Final Answer: [final answer to the human user's question]
When selecting a tool, the AI Agent must provide both the "Tool:" and "Tool Input:" pair in the same response, but on separate lines. "Observation:" marks the beginning of a tool's result, and the AI Agent trusts these results.

The AI Agent must use the conversation_history tool to infer context when necessary.
If a question is vague or requires context, the AI Agent should explicitly use the conversation_history tool with a clear Tool Input focused on finding the relevant context.
The AI Agent should not ask the human user for additional information, clarification, or context.
If the AI Agent cannot find a specific answer after exhausting available tools and approaches, it answers with Final Answer: inconclusive


Question: {query}
Thought:
{transcript}
"""
search_key = os.environ.get("SERPERDEV_API_KEY")
if not search_key:
    raise ValueError("Please set the SERPERDEV_API_KEY environment variable")

openai_key = os.environ.get("OPENAI_API_KEY")
if not search_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

prompt_text = """
Synthesize a comprehensive answer from the following most relevant paragraphs and the given question.
Provide a clear and concise response that summarizes the key points and information presented in the paragraphs.
Your answer should be in your own words and be no longer than 20 words.
\n\n Paragraphs: {documents} \n\n Question: {query} \n\n Answer:
"""

prompt_node = PromptNode(
    "gpt-3.5-turbo",
    default_prompt_template=PromptTemplate("lfqa", prompt_text=prompt_text),
    api_key=openai_key,
    max_length=256,
)

web_retriever = WebRetriever(api_key=search_key, top_search_results=5, mode="snippets")
pipeline = WebQAPipeline(retriever=web_retriever, prompt_node=prompt_node)

few_shot_agent = PromptTemplate("conversational-agent-with-tools", prompt_text=few_shot_prompt)
conversation_history = Tool(
    name="conversation_history",
    pipeline_or_node=lambda tool_input, **kwargs: agent.memory.load(),
    description="useful for when you need to remember what you've already discussed.",
    logging_color=Color.MAGENTA,
)
web_qa_tool = Tool(
    name="Search",
    pipeline_or_node=pipeline,
    description="useful for when you need to Google questions.",
    output_variable="results",
)
pn = PromptNode(
    "gpt-3.5-turbo",
    api_key=os.environ.get("OPENAI_API_KEY"),
    max_length=256,
    stop_words=["Observation:"],
    model_kwargs={"temperature": 0.5, "top_p": 0.9},
)
agent = ConversationalAgentWithTools(
    pn,
    max_steps=10,
    prompt_template=few_shot_agent,
    tools_manager=ToolsManager(tools=[web_qa_tool, conversation_history]),
    memory=ConversationSummaryMemory(pn, summary_frequency=1),
)

test = False
if test:
    questions = [
        "Why was Jamie Foxx recently hospitalized?",
        "Where was he hospitalized?",
        "What movie was he filming at the time?",
        "Who is his female co-star in Back in Action?",
        "Tell me more about her, who is her partner?",
    ]
    for question in questions:
        print(f"\nHuman: {question}")
        agent.run(question)
else:
    while True:
        user_input = input("\nHuman (type 'exit' or 'quit' to quit): ")
        if user_input.lower() == "exit" or user_input.lower() == "quit":
            break
        if user_input.lower() == "memory":
            print("\nMemory:\n", agent.memory.load())
        else:
            assistant_response = agent.run(user_input)
