from unittest.mock import patch

import pytest

from haystack.cli.entry_point import main_cli
from haystack.nodes.prompt.prompt_template import PromptNotFoundError


@pytest.mark.unit
@patch("haystack.cli.prompt.fetch.fetch_from_prompthub")
@patch("haystack.cli.prompt.fetch.cache_prompt")
def test_prompt_fetch_no_args(mock_cache, mock_fetch, cli_runner):
    response = cli_runner.invoke(main_cli, ["prompt", "fetch"])
    assert response.exit_code == 0

    mock_fetch.assert_not_called()
    mock_cache.assert_not_called()


@pytest.mark.unit
@patch("haystack.cli.prompt.fetch.fetch_from_prompthub")
@patch("haystack.cli.prompt.fetch.cache_prompt")
def test_prompt_fetch(mock_cache, mock_fetch, cli_runner):
    response = cli_runner.invoke(main_cli, ["prompt", "fetch", "deepset/question-generation"])
    assert response.exit_code == 0

    mock_fetch.assert_called_once_with("deepset/question-generation")
    mock_cache.assert_called_once()


@pytest.mark.unit
@patch("haystack.cli.prompt.fetch.fetch_from_prompthub")
@patch("haystack.cli.prompt.fetch.cache_prompt")
def test_prompt_fetch_with_multiple_prompts(mock_cache, mock_fetch, cli_runner):
    response = cli_runner.invoke(
        main_cli, ["prompt", "fetch", "deepset/question-generation", "deepset/conversational-agent"]
    )
    assert response.exit_code == 0

    assert mock_fetch.call_count == 2
    mock_fetch.assert_any_call("deepset/question-generation")
    mock_fetch.assert_any_call("deepset/conversational-agent")

    assert mock_cache.call_count == 2


@pytest.mark.unit
@patch("haystack.cli.prompt.fetch.fetch_from_prompthub")
@patch("haystack.cli.prompt.fetch.cache_prompt")
def test_prompt_fetch_with_unexisting_prompt(mock_cache, mock_fetch, cli_runner):
    prompt_name = "deepset/martian-speak"
    error_message = f"Prompt template named '{prompt_name}' not available in the Prompt Hub."
    mock_fetch.side_effect = PromptNotFoundError(error_message)

    response = cli_runner.invoke(main_cli, ["prompt", "fetch", prompt_name])
    assert response.exit_code == 1
    assert error_message in response.output

    mock_fetch.assert_called_once_with(prompt_name)
    mock_cache.assert_not_called()
