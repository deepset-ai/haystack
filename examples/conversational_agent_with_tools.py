import os

from haystack.agents.base import ConversationalAgentWithTools, Tool, ToolsManager
from haystack.agents.memory import ConversationSummaryMemory
from haystack.nodes import PromptNode, WebRetriever, PromptTemplate
from haystack.pipelines import WebQAPipeline

few_shot_prompt = """
Answer questions as accurately as possible. If you're unsure or if the information might be outdated or inaccurate,
use the available tools when necessary to find the most up-to-date and accurate information. The available tools are:

{tool_names_with_descriptions}

Always follow the steps below to answer a question. Each step starts with either a Thought, Tool, Tool Input,
Observation, or Final Answer. Trust the results of the tools, don't try to second-guess them.

Example 1:
Question: What is the capital of France?
Thought: This is a common knowledge question, so I'll answer it directly.
Final Answer: The capital of France is Paris.

Example 2: Who was the first president of the United States?
Thought: This is a common knowledge question, so I'll answer it directly.
Final Answer: The first president of the United States was George Washington.

Example 3:
Question: What is the latest version of Python programming language?
Thought: As of my knowledge cutoff date in September 2021, I'll use the search plugin to help me answer the question.
Tool: Search
Tool Input: What is the latest version of Python programming language as of today?
Observation: 3.11.2
Thought: According to Search Tool, the latest Python version is 3.11.2, now I can answer the question.
Final Answer: The latest Python version is 3.11.2.

Example 4:
Question: How did life evolve from nonliving matter?
Thought: This is a complex question, so I'll use the search plugin to help me answer the question.
Tool: Search
Tool Input: How did life evolve from nonliving matter?
Observation: MIT physicist Dr. Jeremy England recently put forward a theory that attempts to explain the origin of life in terms of fundamental principles of physics. In this view, life is the inevitable result of rising entropy.
Thought: Search Tool has provided me with additional information to what I already know, therefore I'll now use to answer the question.
Final Answer: If life is the inevitable result of rising entropy then the arrival of life should not be a surprise. The idea is highly speculative. Recent computer simulations, however, may be lending support to it. The simulations show that ordinary chemical reactions (of the sort that would have been common on the newly formed Earth) can lead to the creation of highly structured compounds — seemingly a crucial stepping-stone on the path to living organisms.

Users sometimes ask questions that are unclear or lack context. In such cases consults with the following
conversational history to better understand the question context:
{history}

Now, please answer the following question:
Question: {query}
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
Your answer should be in your own words and be no longer than 50 words.
\n\n Paragraphs: {documents} \n\n Question: {query} \n\n Answer:
"""

prompt_node = PromptNode(
    "gpt-3.5-turbo",
    default_prompt_template=PromptTemplate("lfqa", prompt_text=prompt_text),
    api_key=openai_key,
    max_length=256,
)

web_retriever = WebRetriever(api_key=search_key, top_search_results=2, mode="preprocessed_documents")
pipeline = WebQAPipeline(retriever=web_retriever, prompt_node=prompt_node)

few_shot_agent = PromptTemplate("conversational-agent-with-tools", prompt_text=few_shot_prompt)
web_qa_tool = Tool(
    name="Search",
    pipeline_or_node=pipeline,
    description="useful for when you need to Google questions.",
    output_variable="results",
)
pn = PromptNode("gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY"), max_length=256, stop_words=["Observation:"])
agent = ConversationalAgentWithTools(
    pn,
    prompt_template=few_shot_agent,
    tools_manager=ToolsManager(tools=[web_qa_tool]),
    memory=ConversationSummaryMemory(pn),
)

while True:
    user_input = input("Human (type 'exit' or 'quit' to quit): ")
    if user_input.lower() == "exit" or user_input.lower() == "quit":
        break
    elif user_input.lower() == "memory":
        print("\nMemory:\n", agent.memory.load())
    else:
        assistant_response = agent.run(user_input)
        print("\nAssistant:", assistant_response)
