from haystack.agents.agent_step import AgentStep

class ToolResultsGatherer:
    results = []

    def __init__(self, agent_callback_manager, tool_manager_callback_manager):
        agent_callback_manager.on_agent_start += self.on_agent_start
        agent_callback_manager.on_agent_finish += self.on_agent_finish

        tool_manager_callback_manager.on_tool_finish += self.on_tool_finish

    def on_agent_start(self, name, query, params):
        self.results = []

    def on_tool_finish(self, tool_result: str, observation_prefix, llm_prefix, color, tool_name: str, unprocessed_tool_result):
        self.results.append({
            "name": tool_name,
            "result": tool_result,
            "unprocessed_result": unprocessed_tool_result
        })

    def on_agent_finish(self, agent_step: AgentStep):
        agent_step.tool_results = self.results