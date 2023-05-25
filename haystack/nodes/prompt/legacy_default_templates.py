from haystack.nodes.prompt.prompt_template import PromptTemplate
from haystack.nodes.prompt.shapers import AnswerParser

#############################################################################
# This templates were hardcoded in the prompt_template module. When adding
# support for PromptHub integration we decided to remove them with the PR
# that added the integration: https://github.com/deepset-ai/haystack/pull/4879/
#
# That PR also changed the PromptNode API forcing the user to change how
# they use the node.
#
# After some discussion we deemed the change to be too breaking for existing
# use cases and which steps would have been necessary to migrate to the
# new API in case someone was using an harcoded template we decided to
# bring them back.
#
# So for the time being this must live here, no new template must be added
# to this dictionary.
#############################################################################
LEGACY_DEFAULT_TEMPLATES = {
    # DO NOT ADD ANY NEW TEMPLATE IN HERE!
    "question-answering": PromptTemplate(
        prompt="Given the context please answer the question. Context: {join(documents)}; Question: "
        "{query}; Answer:",
        output_parser=AnswerParser(),
    ),
    "question-answering-per-document": PromptTemplate(
        prompt="Given the context please answer the question. Context: {documents}; Question: " "{query}; Answer:",
        output_parser=AnswerParser(),
    ),
    "question-answering-with-references": PromptTemplate(
        prompt="Create a concise and informative answer (no more than 50 words) for a given question "
        "based solely on the given documents. You must only use information from the given documents. "
        "Use an unbiased and journalistic tone. Do not repeat text. Cite the documents using Document[number] notation. "
        "If multiple documents contain the answer, cite those documents like ‘as stated in Document[number], Document[number], etc.’. "
        "If the documents do not contain the answer to the question, say that ‘answering is not possible given the available information.’\n"
        "{join(documents, delimiter=new_line, pattern=new_line+'Document[$idx]: $content', str_replace={new_line: ' ', '[': '(', ']': ')'})} \n Question: {query}; Answer: ",
        output_parser=AnswerParser(reference_pattern=r"Document\[(\d+)\]"),
    ),
    "question-answering-with-document-scores": PromptTemplate(
        prompt="Answer the following question using the paragraphs below as sources. "
        "An answer should be short, a few words at most.\n"
        "Paragraphs:\n{documents}\n"
        "Question: {query}\n\n"
        "Instructions: Consider all the paragraphs above and their corresponding scores to generate "
        "the answer. While a single paragraph may have a high score, it's important to consider all "
        "paragraphs for the same answer candidate to answer accurately.\n\n"
        "After having considered all possibilities, the final answer is:\n"
    ),
    "question-generation": PromptTemplate(
        prompt="Given the context please generate a question. Context: {documents}; Question:"
    ),
    "conditioned-question-generation": PromptTemplate(
        prompt="Please come up with a question for the given context and the answer. "
        "Context: {documents}; Answer: {answers}; Question:"
    ),
    "summarization": PromptTemplate(prompt="Summarize this document: {documents} Summary:"),
    "question-answering-check": PromptTemplate(
        prompt="Does the following context contain the answer to the question? "
        "Context: {documents}; Question: {query}; Please answer yes or no! Answer:",
        output_parser=AnswerParser(),
    ),
    "sentiment-analysis": PromptTemplate(
        prompt="Please give a sentiment for this context. Answer with positive, "
        "negative or neutral. Context: {documents}; Answer:"
    ),
    "multiple-choice-question-answering": PromptTemplate(
        prompt="Question:{query} ; Choose the most suitable option to answer the above question. "
        "Options: {options}; Answer:",
        output_parser=AnswerParser(),
    ),
    "topic-classification": PromptTemplate(
        prompt="Categories: {options}; What category best describes: {documents}; Answer:"
    ),
    "language-detection": PromptTemplate(
        prompt="Detect the language in the following context and answer with the "
        "name of the language. Context: {documents}; Answer:"
    ),
    "translation": PromptTemplate(
        prompt="Translate the following context to {target_language}. Context: {documents}; Translation:"
    ),
    "zero-shot-react": PromptTemplate(
        prompt="You are a helpful and knowledgeable agent. To achieve your goal of answering complex questions "
        "correctly, you have access to the following tools:\n\n"
        "{tool_names_with_descriptions}\n\n"
        "To answer questions, you'll need to go through multiple steps involving step-by-step thinking and "
        "selecting appropriate tools and their inputs; tools will respond with observations. When you are ready "
        "for a final answer, respond with the `Final Answer:`\n\n"
        "Use the following format:\n\n"
        "Question: the question to be answered\n"
        "Thought: Reason if you have the final answer. If yes, answer the question. If not, find out the missing information needed to answer it.\n"
        "Tool: pick one of {tool_names} \n"
        "Tool Input: the input for the tool\n"
        "Observation: the tool will respond with the result\n"
        "...\n"
        "Final Answer: the final answer to the question, make it short (1-5 words)\n\n"
        "Thought, Tool, Tool Input, and Observation steps can be repeated multiple times, but sometimes we can find an answer in the first pass\n"
        "---\n\n"
        "Question: {query}\n"
        "Thought: Let's think step-by-step, I first need to {transcript}"
    ),
    "conversational-agent": PromptTemplate(
        prompt="The following is a conversation between a human and an AI.\n{history}\nHuman: {query}\nAI:"
    ),
    "conversational-summary": PromptTemplate(
        prompt="Condense the following chat transcript by shortening and summarizing the content without losing important information:\n{chat_transcript}\nCondensed Transcript:"
    ),
    # DO NOT ADD ANY NEW TEMPLATE IN HERE!
}
