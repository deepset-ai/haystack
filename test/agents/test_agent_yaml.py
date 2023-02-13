import pytest

from haystack.agents import Agent
from test.conftest import SAMPLES_PATH


@pytest.mark.skip
@pytest.mark.integration
def test_load_and_save_from_yaml(tmp_path):
    config_path = SAMPLES_PATH / "agent" / "test.haystack-agent.yml"

    search_and_calculate_agent = Agent.load_from_yaml(path=config_path, agent_name="search_and_calculate")

    search_and_calculate_agent.load_from_yaml(path=config_path, agent_name="search_and_calculate")

    new_agent_config = tmp_path / "test_agent.yaml"
    search_and_calculate_agent.save_to_yaml(new_agent_config)
