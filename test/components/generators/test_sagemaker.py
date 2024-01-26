from typing import List

import os
from unittest.mock import patch, Mock

import pytest
from openai import OpenAIError

from haystack.components.generators.sagemaker import SagemakerGenerator


class TestSagemakerGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")

        component = SagemakerGenerator(model="test-model")
        assert component.model == "test-model"
        assert component.aws_access_key_id_var == "AWS_ACCESS_KEY_ID"
        assert component.aws_secret_access_key_var == "AWS_SECRET_ACCESS_KEY"
        assert component.aws_session_token_var == "AWS_SESSION_TOKEN"
        assert component.aws_region_name_var == "AWS_REGION"
        assert component.aws_profile_name_var == "AWS_PROFILE"
        assert component.aws_custom_attributes == {}
        assert component.generation_kwargs == {"max_new_tokens": 1024}
        assert component.client is None

    def test_init_fail_wo_access_key_or_secret_key(self, monkeypatch):
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        with pytest.raises(ValueError):
            SagemakerGenerator(model="test-model")

        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        with pytest.raises(ValueError):
            SagemakerGenerator(model="test-model")

        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
        with pytest.raises(ValueError):
            SagemakerGenerator(model="test-model")

    def test_init_with_parameters(self, monkeypatch):
        monkeypatch.setenv("MY_ACCESS_KEY_ID", "test-access-key")
        monkeypatch.setenv("MY_SECRET_ACCESS_KEY", "test-secret-key")

        component = SagemakerGenerator(
            model="test-model",
            aws_access_key_id_var="MY_ACCESS_KEY_ID",
            aws_secret_access_key_var="MY_SECRET_ACCESS_KEY",
            aws_session_token_var="MY_SESSION_TOKEN",
            aws_region_name_var="MY_REGION",
            aws_profile_name_var="MY_PROFILE",
            aws_custom_attributes={"custom": "attr"},
            generation_kwargs={"generation": "kwargs"},
        )
        assert component.model == "test-model"
        assert component.aws_access_key_id_var == "MY_ACCESS_KEY_ID"
        assert component.aws_secret_access_key_var == "MY_SECRET_ACCESS_KEY"
        assert component.aws_session_token_var == "MY_SESSION_TOKEN"
        assert component.aws_region_name_var == "MY_REGION"
        assert component.aws_profile_name_var == "MY_PROFILE"
        assert component.aws_custom_attributes == {"custom": "attr"}
        assert component.generation_kwargs == {"generation": "kwargs"}
        assert component.client is None

    def test_run(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
        client_mock = Mock()
        client_mock.invoke_endpoint.return_value = {
            "Body": Mock(read=lambda: b'[{"generated_text": "test-reply", "other": "metadata"}]')
        }

        component = SagemakerGenerator(model="test-model")
        component.client = client_mock  # Simulate warm_up()
        response = component.run("What's Natural Language Processing?")

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, str) for reply in response["replies"]]
        assert "test-reply" in response["replies"][0]

        assert "meta" in response
        assert isinstance(response["meta"], list)
        assert len(response["meta"]) == 1
        assert [isinstance(reply, dict) for reply in response["meta"]]
        assert response["meta"][0]["other"] == "metadata"

    @pytest.mark.skipif(
        (
            not os.environ.get("AWS_ACCESS_KEY_ID", None)
            or not os.environ.get("AWS_SECRET_ACCESS_KEY", None)
            or not os.environ.get("AWS_SAGEMAKER_TEST_MODEL", None)
        ),
        reason="Export two env vars called AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY containing the AWS credentials to run this test.",
    )
    @pytest.mark.integration
    def test_run_falcon(self):
        component = SagemakerGenerator(
            model=os.getenv("AWS_SAGEMAKER_TEST_MODEL"), generation_kwargs={"max_new_tokens": 10}
        )
        component.warm_up()
        response = component.run("What's Natural Language Processing?")

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, str) for reply in response["replies"]]

        # Coarse check: assuming no more than 4 chars per token. In any case it
        # will fail if the `max_new_tokens` parameter is not respected, as the
        # default is either 256 or 1024
        assert all(len(reply) <= 40 for reply in response["replies"])

        assert "meta" in response
        assert isinstance(response["meta"], list)
        assert len(response["meta"]) == 1
        assert [isinstance(reply, dict) for reply in response["meta"]]
