import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock

import pytest
import yaml

from haystack.utils.deepsetcloud import DeepsetCloudClient, PipelineClient


@pytest.fixture
def pipeline_config(samples_path: Path) -> Dict[str, Any]:
    with (samples_path / "dc" / "pipeline_config.json").open() as f:
        return json.load(f)


@pytest.fixture()
def mocked_client() -> Mock:
    api_client = Mock(spec=DeepsetCloudClient)

    api_client.build_workspace_url.return_value = "https://dc"

    return api_client


@pytest.fixture()
def mock_success_response() -> Mock:
    mock_response = Mock()
    mock_response.json.return_value = {"name": "test_pipeline"}

    return mock_response


class TestSaveConfig:
    def test_save_config(
        self, pipeline_config: Dict[str, Any], mocked_client: Mock, mock_success_response: Mock
    ) -> None:
        mocked_client.post.return_value = mock_success_response

        pipeline_name = "test_pipeline"
        workspace_name = "test_workspace"

        pipeline_client = PipelineClient(client=mocked_client)

        pipeline_client.save_pipeline_config(
            config=pipeline_config, pipeline_config_name=pipeline_name, workspace=workspace_name
        )

        expected_payload = {"name": pipeline_name, "config": yaml.dump(pipeline_config)}
        mocked_client.post.assert_called_once_with(url="https://dc/pipelines", json=expected_payload, headers=None)


class TestUpdateConfig:
    def test_update_config(
        self, pipeline_config: Dict[str, Any], mocked_client: Mock, mock_success_response: Mock
    ) -> None:
        mocked_client.put.return_value = mock_success_response
        pipeline_name = "test_pipeline"
        workspace_name = "test_workspace"

        pipeline_client = PipelineClient(client=mocked_client)

        pipeline_client.update_pipeline_config(
            config=pipeline_config, pipeline_config_name=pipeline_name, workspace=workspace_name
        )

        mocked_client.put.assert_called_once_with(
            url=f"https://dc/pipelines/{pipeline_name}/yaml", data=yaml.dump(pipeline_config), headers=None
        )
