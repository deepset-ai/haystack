# pylint: disable=missing-timeout

import json
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import logging
import os
import time
from enum import Enum

import pandas as pd
import requests
import yaml
from tqdm.auto import tqdm

from haystack.schema import Answer, Document, EvaluationResult, FilterType, Label

DEFAULT_API_ENDPOINT = "https://api.cloud.deepset.ai/api/v1"


class PipelineStatus(Enum):
    UNDEPLOYED: str = "UNDEPLOYED"
    DEPLOYED_UNHEALTHY: str = "DEPLOYED_UNHEALTHY"
    DEPLOYED: str = "DEPLOYED"
    DEPLOYMENT_IN_PROGRESS: str = "DEPLOYMENT_IN_PROGRESS"
    UNDEPLOYMENT_IN_PROGRESS: str = "UNDEPLOYMENT_IN_PROGRESS"
    DEPLOYMENT_SCHEDULED: str = "DEPLOYMENT_SCHEDULED"
    UNDEPLOYMENT_SCHEDULED: str = "UNDEPLOYMENT_SCHEDULED"
    DEPLOYMENT_FAILED: str = "DEPLOYMENT_FAILED"
    UNDEPLOYMENT_FAILED: str = "UNDEPLOYMENT_FAILED"
    UKNOWN: str = "UNKNOWN"

    @classmethod
    def from_str(cls, status_string: str) -> "PipelineStatus":
        return cls.__dict__.get(status_string, PipelineStatus.UKNOWN)


SATISFIED_STATES_KEY = "satisfied_states"
FAILED_STATES_KEY = "failed_states"
VALID_INITIAL_STATES_KEY = "valid_initial_states"
VALID_TRANSITIONING_STATES_KEY = "valid_transitioning_states"
PIPELINE_STATE_TRANSITION_INFOS: Dict[PipelineStatus, Dict[str, List[PipelineStatus]]] = {
    PipelineStatus.UNDEPLOYED: {
        SATISFIED_STATES_KEY: [PipelineStatus.UNDEPLOYED],
        FAILED_STATES_KEY: [PipelineStatus.UNDEPLOYMENT_FAILED],
        VALID_INITIAL_STATES_KEY: [
            PipelineStatus.DEPLOYED,
            PipelineStatus.DEPLOYMENT_FAILED,
            PipelineStatus.UNDEPLOYMENT_FAILED,
        ],
        VALID_TRANSITIONING_STATES_KEY: [
            PipelineStatus.UNDEPLOYMENT_SCHEDULED,
            PipelineStatus.UNDEPLOYMENT_IN_PROGRESS,
        ],
    },
    PipelineStatus.DEPLOYED: {
        SATISFIED_STATES_KEY: [PipelineStatus.DEPLOYED, PipelineStatus.DEPLOYED_UNHEALTHY],
        FAILED_STATES_KEY: [PipelineStatus.DEPLOYMENT_FAILED],
        VALID_INITIAL_STATES_KEY: [
            PipelineStatus.UNDEPLOYED,
            PipelineStatus.DEPLOYMENT_FAILED,
            PipelineStatus.UNDEPLOYMENT_FAILED,
        ],
        VALID_TRANSITIONING_STATES_KEY: [PipelineStatus.DEPLOYMENT_SCHEDULED, PipelineStatus.DEPLOYMENT_IN_PROGRESS],
    },
}

logger = logging.getLogger(__name__)


class BearerAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token

    def __call__(self, r):
        r.headers["authorization"] = "Bearer " + self.token
        return r


class DeepsetCloudError(Exception):
    """Raised when there is an error communicating with deepset Cloud"""


class DeepsetCloudClient:
    def __init__(self, api_key: Optional[str] = None, api_endpoint: Optional[str] = None):
        """
        A client to communicate with deepset Cloud.

        :param api_key: Secret value of the API key.
                        If not specified, it's read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the deepset Cloud API.
                             If not specified, it's read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
                             If environment variable is not set, defaults to 'https://api.cloud.deepset.ai/api/v1'.
        """
        self.api_key = api_key or os.getenv("DEEPSET_CLOUD_API_KEY")
        if self.api_key is None:
            raise DeepsetCloudError(
                "No api_key specified. Please set api_key param or DEEPSET_CLOUD_API_KEY environment variable."
            )

        self.api_endpoint = api_endpoint or os.getenv("DEEPSET_CLOUD_API_ENDPOINT", DEFAULT_API_ENDPOINT)

    def get(
        self,
        url: str,
        query_params: Optional[dict] = None,
        headers: Optional[dict] = None,
        stream: bool = False,
        raise_on_error: bool = True,
    ):
        return self._execute_request(
            method="GET",
            url=url,
            query_params=query_params,
            headers=headers,
            stream=stream,
            raise_on_error=raise_on_error,
        )

    def get_with_auto_paging(
        self,
        url: str,
        query_params: Optional[dict] = None,
        headers: Optional[dict] = None,
        stream: bool = False,
        raise_on_error: bool = True,
        auto_paging_page_size: Optional[int] = None,
    ) -> Generator:
        return self._execute_auto_paging_request(
            method="GET",
            url=url,
            query_params=query_params,
            headers=headers,
            stream=stream,
            raise_on_error=raise_on_error,
            auto_paging_page_size=auto_paging_page_size,
        )

    def post(
        self,
        url: str,
        json: dict = {},
        data: Optional[Any] = None,
        query_params: Optional[dict] = None,
        headers: Optional[dict] = None,
        stream: bool = False,
        files: Optional[Any] = None,
        raise_on_error: bool = True,
    ):
        return self._execute_request(
            method="POST",
            url=url,
            query_params=query_params,
            json=json,
            data=data,
            stream=stream,
            files=files,
            headers=headers,
            raise_on_error=raise_on_error,
        )

    def post_with_auto_paging(
        self,
        url: str,
        json: dict = {},
        data: Optional[Any] = None,
        query_params: Optional[dict] = None,
        headers: Optional[dict] = None,
        stream: bool = False,
        raise_on_error: bool = True,
        auto_paging_page_size: Optional[int] = None,
    ):
        return self._execute_auto_paging_request(
            method="POST",
            url=url,
            query_params=query_params,
            json=json,
            data=data,
            stream=stream,
            headers=headers,
            raise_on_error=raise_on_error,
            auto_paging_page_size=auto_paging_page_size,
        )

    def put(
        self,
        url: str,
        json: Optional[dict] = None,
        data: Optional[Any] = None,
        query_params: Optional[dict] = None,
        stream: bool = False,
        headers: Optional[dict] = None,
        raise_on_error: bool = True,
    ):
        return self._execute_request(
            method="PUT",
            url=url,
            query_params=query_params,
            json=json,
            data=data,
            stream=stream,
            headers=headers,
            raise_on_error=raise_on_error,
        )

    def put_with_auto_paging(
        self,
        url: str,
        json: dict = {},
        data: Optional[Any] = None,
        query_params: Optional[dict] = None,
        headers: Optional[dict] = None,
        stream: bool = False,
        raise_on_error: bool = True,
        auto_paging_page_size: Optional[int] = None,
    ):
        return self._execute_auto_paging_request(
            method="PUT",
            url=url,
            query_params=query_params,
            json=json,
            data=data,
            stream=stream,
            headers=headers,
            raise_on_error=raise_on_error,
            auto_paging_page_size=auto_paging_page_size,
        )

    def delete(
        self,
        url: str,
        query_params: Optional[dict] = None,
        headers: Optional[dict] = None,
        stream: bool = False,
        raise_on_error: bool = True,
    ):
        return self._execute_request(
            method="DELETE",
            url=url,
            query_params=query_params,
            headers=headers,
            stream=stream,
            raise_on_error=raise_on_error,
        )

    def patch(
        self,
        url: str,
        json: Optional[dict] = None,
        data: Optional[Any] = None,
        query_params: Optional[dict] = None,
        stream: bool = False,
        headers: Optional[dict] = None,
        raise_on_error: bool = True,
    ):
        return self._execute_request(
            method="PATCH",
            url=url,
            query_params=query_params,
            json=json,
            data=data,
            stream=stream,
            headers=headers,
            raise_on_error=raise_on_error,
        )

    def _execute_auto_paging_request(
        self,
        method: Literal["GET", "POST", "PUT", "HEAD", "DELETE"],
        url: str,
        json: Optional[dict] = None,
        data: Optional[Any] = None,
        query_params: Optional[dict] = None,
        headers: Optional[dict] = None,
        stream: bool = False,
        raise_on_error: bool = True,
        auto_paging_page_size: Optional[int] = None,
    ) -> Generator:
        query_params = query_params.copy() if query_params is not None else {}
        if auto_paging_page_size:
            query_params["limit"] = auto_paging_page_size
        page_number = 1
        has_more = True
        while has_more:
            query_params["page_number"] = page_number
            payload = self._execute_request(
                method=method,
                url=url,
                json=json,
                data=data,
                query_params=query_params,
                headers=headers,
                stream=stream,
                raise_on_error=raise_on_error,
            ).json()
            yield from payload["data"]
            has_more = payload["has_more"]
            page_number += 1

    def _execute_request(
        self,
        method: Literal["GET", "POST", "PUT", "HEAD", "DELETE", "PATCH"],
        url: str,
        json: Optional[dict] = None,
        data: Optional[Any] = None,
        query_params: Optional[dict] = None,
        headers: Optional[dict] = None,
        stream: bool = False,
        files: Optional[Any] = None,
        raise_on_error: bool = True,
    ):
        if json is not None:
            json = self._remove_null_values(json)
        response = requests.request(
            method=method,
            url=url,
            json=json,
            data=data,
            params=query_params,
            headers=headers,
            auth=BearerAuth(self.api_key),
            stream=stream,
            files=files,
        )
        if raise_on_error and response.status_code > 299:
            raise DeepsetCloudError(
                f"{method} {url} failed: HTTP {response.status_code} - {response.reason}\n{response.content.decode()}"
            )
        return response

    def build_workspace_url(self, workspace: Optional[str] = None):
        api_endpoint = f"{self.api_endpoint}".rstrip("/")
        url = f"{api_endpoint}/workspaces/{workspace}"
        return url

    def _remove_null_values(self, body: dict) -> dict:
        return {k: v for k, v in body.items() if v is not None}


class IndexClient:
    def __init__(self, client: DeepsetCloudClient, workspace: Optional[str] = None, index: Optional[str] = None):
        """
        A client to communicate with deepset Cloud indexes.

        :param client: deepset Cloud client
        :param workspace: Specifies the name of the workspace for which you want to create the client.
        :param index: index in deepset Cloud workspace

        """
        self.client = client
        self.workspace = workspace
        self.index = index

    def info(self, workspace: Optional[str] = None, index: Optional[str] = None, headers: Optional[dict] = None):
        index_url = self._build_index_url(workspace=workspace, index=index)
        try:
            response = self.client.get(url=index_url, headers=headers)
            return response.json()
        except Exception as ie:
            raise DeepsetCloudError(f"Could not connect to deepset Cloud:\n{ie}") from ie

    def query(
        self,
        query: Optional[str] = None,
        filters: Optional[FilterType] = None,
        top_k: int = 10,
        custom_query: Optional[str] = None,
        query_emb: Optional[List[float]] = None,
        return_embedding: Optional[bool] = None,
        workspace: Optional[str] = None,
        index: Optional[str] = None,
        all_terms_must_match: Optional[bool] = None,
        scale_score: bool = True,
        headers: Optional[dict] = None,
    ) -> List[dict]:
        index_url = self._build_index_url(workspace=workspace, index=index)
        query_url = f"{index_url}/documents-query"
        request = {
            "query": query,
            "filters": filters,
            "top_k": top_k,
            "custom_query": custom_query,
            "query_emb": query_emb,
            "return_embedding": return_embedding,
            "all_terms_must_match": all_terms_must_match,
            "scale_score": scale_score,
        }
        response = self.client.post(url=query_url, json=request, headers=headers)
        return response.json()

    def stream_documents(
        self,
        return_embedding: Optional[bool] = False,
        filters: Optional[FilterType] = None,
        workspace: Optional[str] = None,
        index: Optional[str] = None,
        headers: Optional[dict] = None,
    ):
        index_url = self._build_index_url(workspace=workspace, index=index)
        query_url = f"{index_url}/documents-stream"
        request = {"return_embedding": return_embedding, "filters": filters}
        response = self.client.post(url=query_url, json=request, headers=headers, stream=True)
        return response.iter_lines()

    def get_document(
        self, id: str, workspace: Optional[str] = None, index: Optional[str] = None, headers: Optional[dict] = None
    ):
        index_url = self._build_index_url(workspace=workspace, index=index)
        document_url = f"{index_url}/documents/{id}"
        response = self.client.get(url=document_url, headers=headers, raise_on_error=False)
        doc: Optional[dict] = None
        if response.status_code == 200:
            doc = response.json()
        else:
            logger.warning(
                f"Document {id} could not be fetched from deepset Cloud: HTTP {response.status_code} - {response.reason}\n{response.content.decode()}"
            )
        return doc

    def count_documents(
        self,
        filters: Optional[FilterType] = None,
        only_documents_without_embedding: Optional[bool] = False,
        workspace: Optional[str] = None,
        index: Optional[str] = None,
        headers: Optional[dict] = None,
    ) -> dict:
        index_url = self._build_index_url(workspace=workspace, index=index)
        count_url = f"{index_url}/documents-count"
        request = {"filters": filters, "only_documents_without_embedding": only_documents_without_embedding}
        response = self.client.post(url=count_url, json=request, headers=headers)
        return response.json()

    def _build_index_url(self, workspace: Optional[str] = None, index: Optional[str] = None):
        if workspace is None:
            workspace = self.workspace
        if index is None:
            index = self.index
        workspace_url = self.client.build_workspace_url(workspace)
        return f"{workspace_url}/indexes/{index}"


class PipelineClient:
    def __init__(
        self, client: DeepsetCloudClient, workspace: Optional[str] = None, pipeline_config_name: Optional[str] = None
    ):
        """
        A client to communicate with deepset Cloud pipelines.

        :param client: deepset Cloud client
        :param workspace: Specifies the name of the workspace for which you want to create the client.
        :param pipeline_config_name: Name of the pipeline_config in deepset Cloud workspace.

        """
        self.client = client
        self.workspace = workspace
        self.pipeline_config_name = pipeline_config_name

    def get_pipeline_config(
        self,
        workspace: Optional[str] = None,
        pipeline_config_name: Optional[str] = None,
        headers: Optional[dict] = None,
    ) -> dict:
        """
        Gets the config from a pipeline on deepset Cloud.

        :param pipeline_config_name: Name of the pipeline_config in deepset Cloud workspace.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
        :param headers: Headers to pass to the API call.
        """
        pipeline_url = self._build_pipeline_url(workspace=workspace, pipeline_config_name=pipeline_config_name)
        pipeline_config_url = f"{pipeline_url}/json"
        response = self.client.get(url=pipeline_config_url, headers=headers).json()
        return response

    def get_pipeline_config_info(
        self,
        workspace: Optional[str] = None,
        pipeline_config_name: Optional[str] = None,
        headers: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Gets information about a pipeline on deepset Cloud.

        :param pipeline_config_name: Name of the pipeline_config in deepset Cloud workspace.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
        :param headers: Headers to pass to the API call.
        """
        pipeline_url = self._build_pipeline_url(workspace=workspace, pipeline_config_name=pipeline_config_name)
        response = self.client.get(url=pipeline_url, headers=headers, raise_on_error=False)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
        else:
            raise DeepsetCloudError(
                f"GET {pipeline_url} failed: HTTP {response.status_code} - {response.reason}\n{response.content.decode()}"
            )

    def list_pipeline_configs(self, workspace: Optional[str] = None, headers: Optional[dict] = None) -> Generator:
        """
        Lists all pipelines available on deepset Cloud.

        :param workspace: Specifies the name of the workspace on deepset Cloud.
        :param headers: Headers to pass to the API call.

        Returns:
            Generator of dictionaries: List[dict]
            each dictionary: {
                        "name": str -> `pipeline_config_name` to be used in `load_from_deepset_cloud()`,
                        "..." -> additional pipeline meta information
                        }
            example:

            ```python
            [{'name': 'my_super_nice_pipeline_config',
                'pipeline_id': '2184e0c1-c6ec-40a1-9b28-5d2768e5efa2',
                'status': 'DEPLOYED',
                'created_at': '2022-02-01T09:57:03.803991+00:00',
                'deleted': False,
                'is_default': False,
                'indexing': {'status': 'IN_PROGRESS',
                'pending_file_count': 3,
                'total_file_count': 31}}]
            ```

        """
        workspace_url = self._build_workspace_url(workspace)
        pipelines_url = f"{workspace_url}/pipelines"
        generator = self.client.get_with_auto_paging(url=pipelines_url, headers=headers)
        return generator

    def save_pipeline_config(
        self,
        config: dict,
        pipeline_config_name: Optional[str] = None,
        workspace: Optional[str] = None,
        headers: Optional[dict] = None,
    ):
        """
        Saves a pipeline config to deepset Cloud.

        :param config: The pipeline config to save.
        :param pipeline_config_name: Name of the pipeline_config in deepset Cloud workspace.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
        :param headers: Headers to pass to the API call.
        """
        config["name"] = pipeline_config_name
        workspace_url = self._build_workspace_url(workspace=workspace)
        pipelines_url = f"{workspace_url}/pipelines"
        response = self.client.post(url=pipelines_url, data=yaml.dump(config), headers=headers).json()
        if "name" not in response or response["name"] != pipeline_config_name:
            logger.warning("Unexpected response from saving pipeline config: %s", response)

    def update_pipeline_config(
        self,
        config: dict,
        pipeline_config_name: Optional[str] = None,
        workspace: Optional[str] = None,
        headers: Optional[dict] = None,
    ):
        """
        Updates a pipeline config on deepset Cloud.

        :param config: The pipeline config to save.
        :param pipeline_config_name: Name of the pipeline_config in deepset Cloud workspace.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
        :param headers: Headers to pass to the API call.
        """
        config["name"] = pipeline_config_name
        pipeline_url = self._build_pipeline_url(workspace=workspace, pipeline_config_name=pipeline_config_name)
        yaml_url = f"{pipeline_url}/yaml"
        response = self.client.put(url=yaml_url, data=yaml.dump(config), headers=headers).json()
        if "name" not in response or response["name"] != pipeline_config_name:
            logger.warning("Unexpected response from updating pipeline config: %s", response)

    def deploy(
        self,
        pipeline_config_name: Optional[str] = None,
        workspace: Optional[str] = None,
        headers: Optional[dict] = None,
        timeout: int = 60,
        show_curl_message: bool = True,
    ):
        """
        Deploys the pipelines of a pipeline config on deepset Cloud.
        Blocks until pipelines are successfully deployed, deployment failed or timeout exceeds.
        If pipelines are already deployed no action will be taken and an info will be logged.
        If timeout exceeds a TimeoutError will be raised.
        If deployment fails a DeepsetCloudError will be raised.

        :param pipeline_config_name: Name of the config file inside the deepset Cloud workspace.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
        :param headers: Headers to pass to the API call.
        :param timeout: The time in seconds to wait until deployment completes.
                        If the timeout is exceeded an error will be raised.
        :param show_curl_message: Whether to print an additional message after successful deployment showing how to query the pipeline using curl.
        """
        status, changed = self._transition_pipeline_state(
            target_state=PipelineStatus.DEPLOYED,
            timeout=timeout,
            pipeline_config_name=pipeline_config_name,
            workspace=workspace,
            headers=headers,
        )

        if workspace is None:
            workspace = self.workspace
        if pipeline_config_name is None:
            pipeline_config_name = self.pipeline_config_name

        pipeline_url = f"{self.client.api_endpoint}/workspaces/{workspace}/pipelines/{pipeline_config_name}/search"

        if status == PipelineStatus.DEPLOYED:
            if changed:
                logger.info("Pipeline config '%s' successfully deployed.", pipeline_config_name)
            else:
                logger.info("Pipeline config '%s' is already deployed.", pipeline_config_name)
            logger.info(
                f"Search endpoint for pipeline config '{pipeline_config_name}' is up and running for you under {pipeline_url}"
            )
            if show_curl_message:
                curl_cmd = (
                    f"curl -X 'POST' \\\n"
                    f"  '{pipeline_url}' \\\n"
                    f"  -H 'accept: application/json' \\\n"
                    f"  -H 'Authorization: Bearer <INSERT_TOKEN_HERE>' \\\n"
                    f"  -H 'Content-Type: application/json' \\\n"
                    f"  -d '{{\n"
                    f'  "queries": [\n'
                    f'    "Is there an answer to this question?"\n'
                    f"  ]\n"
                    f"}}'"
                )
                logger.info("Try it out using the following curl command:\n%s", curl_cmd)

        elif status == PipelineStatus.DEPLOYMENT_FAILED:
            raise DeepsetCloudError(
                f"Deployment of pipeline config '{pipeline_config_name}' failed. "
                "This might be caused by an exception in deepset Cloud or a runtime error in the pipeline. "
                "You can try to run this pipeline locally first."
            )
        elif status in [PipelineStatus.UNDEPLOYMENT_IN_PROGRESS, PipelineStatus.UNDEPLOYMENT_SCHEDULED]:
            raise DeepsetCloudError(
                f"Deployment of pipeline config '{pipeline_config_name}' aborted. Undeployment was requested."
            )
        elif status == PipelineStatus.UNDEPLOYED:
            raise DeepsetCloudError(f"Deployment of pipeline config '{pipeline_config_name}' failed.")
        else:
            raise DeepsetCloudError(
                f"Deployment of pipeline config '{pipeline_config_name} ended in unexpected status: {status.value}"
            )

    def undeploy(
        self,
        pipeline_config_name: Optional[str] = None,
        workspace: Optional[str] = None,
        headers: Optional[dict] = None,
        timeout: int = 60,
    ):
        """
        Undeploys the pipelines of a pipeline config on deepset Cloud.
        Blocks until pipelines are successfully undeployed, undeployment failed or timeout exceeds.
        If pipelines are already undeployed no action will be taken and an info will be logged.
        If timeout exceeds a TimeoutError will be raised.
        If deployment fails a DeepsetCloudError will be raised.

        :param pipeline_config_name: Name of the config file inside the deepset Cloud workspace.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
        :param headers: Headers to pass to the API call
        :param timeout: The time in seconds to wait until undeployment completes.
                        If the timeout is exceeded an error will be raised.
        """
        status, changed = self._transition_pipeline_state(
            target_state=PipelineStatus.UNDEPLOYED,
            timeout=timeout,
            pipeline_config_name=pipeline_config_name,
            workspace=workspace,
            headers=headers,
        )

        if status == PipelineStatus.UNDEPLOYED:
            if changed:
                logger.info("Pipeline config '%s' successfully undeployed.", pipeline_config_name)
            else:
                logger.info("Pipeline config '%s' is already undeployed.", pipeline_config_name)
        elif status in [PipelineStatus.DEPLOYMENT_IN_PROGRESS, PipelineStatus.DEPLOYMENT_SCHEDULED]:
            raise DeepsetCloudError(
                f"Undeployment of pipeline config '{pipeline_config_name}' aborted. Deployment was requested."
            )
        elif status in [PipelineStatus.DEPLOYED, PipelineStatus.DEPLOYED_UNHEALTHY]:
            raise DeepsetCloudError(f"Undeployment of pipeline config '{pipeline_config_name}' failed.")
        else:
            raise DeepsetCloudError(
                f"Undeployment of pipeline config '{pipeline_config_name} ended in unexpected status: {status.value}"
            )

    def _transition_pipeline_state(
        self,
        target_state: Literal[PipelineStatus.DEPLOYED, PipelineStatus.UNDEPLOYED],
        timeout: int = 60,
        pipeline_config_name: Optional[str] = None,
        workspace: Optional[str] = None,
        headers: Optional[dict] = None,
    ) -> Tuple[PipelineStatus, bool]:
        """
        Transitions the pipeline config state to desired target_state on deepset Cloud.

        :param target_state: The target state of the Pipeline config.
        :param pipeline_config_name: Name of the config file inside the deepset Cloud workspace.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
        :param headers: Headers to pass to the API call
        :param timeout: The time in seconds to wait until undeployment completes.
                        If the timeout is exceeded an error will be raised.
        """
        pipeline_info = self.get_pipeline_config_info(
            pipeline_config_name=pipeline_config_name, workspace=workspace, headers=headers
        )
        if pipeline_info is None:
            raise DeepsetCloudError(f"Pipeline config '{pipeline_config_name}' does not exist.")

        transition_info = PIPELINE_STATE_TRANSITION_INFOS[target_state]
        satisfied_states = transition_info[SATISFIED_STATES_KEY]
        failed_states = transition_info[FAILED_STATES_KEY]
        valid_transitioning_states = transition_info[VALID_TRANSITIONING_STATES_KEY]
        valid_initial_states = transition_info[VALID_INITIAL_STATES_KEY]

        status = PipelineStatus.from_str(pipeline_info["status"])
        if status in satisfied_states:
            return status, False

        if status not in valid_initial_states:
            raise DeepsetCloudError(
                f"Pipeline config '{pipeline_config_name}' is in invalid state '{status.value}' to be transitioned to '{target_state.value}'."
            )

        if status in failed_states:
            logger.warning(
                f"Pipeline config '{pipeline_config_name}' is in a failed state '{status}'. This might be caused by a previous error during (un)deployment. "
                + f"Trying to transition from '{status}' to '{target_state}'..."
            )

        if target_state == PipelineStatus.DEPLOYED:
            res = self._deploy(pipeline_config_name=pipeline_config_name, workspace=workspace, headers=headers)
            status = PipelineStatus.from_str(res["status"])
        elif target_state == PipelineStatus.UNDEPLOYED:
            res = self._undeploy(pipeline_config_name=pipeline_config_name, workspace=workspace, headers=headers)
            status = PipelineStatus.from_str(res["status"])
        else:
            raise NotImplementedError(f"Transitioning to state '{target_state.value}' is not implemented.")

        start_time = time.time()
        while status in valid_transitioning_states:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Transitioning of '{pipeline_config_name}' to state '{target_state.value}' timed out."
                )
            pipeline_info = self.get_pipeline_config_info(
                pipeline_config_name=pipeline_config_name, workspace=workspace, headers=headers
            )
            if pipeline_info is None:
                raise DeepsetCloudError(f"Pipeline config '{pipeline_config_name}' does not exist anymore.")
            status = PipelineStatus.from_str(pipeline_info["status"])
            if status in valid_transitioning_states:
                logger.info("Current status of '%s' is: '%s'", pipeline_config_name, status)
                time.sleep(5)

        return status, True

    def _deploy(
        self,
        pipeline_config_name: Optional[str] = None,
        workspace: Optional[str] = None,
        headers: Optional[dict] = None,
    ) -> dict:
        pipeline_url = self._build_pipeline_url(workspace=workspace, pipeline_config_name=pipeline_config_name)
        deploy_url = f"{pipeline_url}/deploy"
        response = self.client.post(url=deploy_url, headers=headers).json()
        return response

    def _undeploy(
        self,
        pipeline_config_name: Optional[str] = None,
        workspace: Optional[str] = None,
        headers: Optional[dict] = None,
    ) -> dict:
        pipeline_url = self._build_pipeline_url(workspace=workspace, pipeline_config_name=pipeline_config_name)
        undeploy_url = f"{pipeline_url}/undeploy"
        response = self.client.post(url=undeploy_url, headers=headers).json()
        return response

    def _build_pipeline_url(self, workspace: Optional[str] = None, pipeline_config_name: Optional[str] = None):
        if pipeline_config_name is None:
            pipeline_config_name = self.pipeline_config_name
        workspace_url = self._build_workspace_url(workspace)
        return f"{workspace_url}/pipelines/{pipeline_config_name}"

    def _build_workspace_url(self, workspace: Optional[str] = None):
        if workspace is None:
            workspace = self.workspace
        return self.client.build_workspace_url(workspace)


class EvaluationSetClient:
    def __init__(
        self, client: DeepsetCloudClient, workspace: Optional[str] = None, evaluation_set: Optional[str] = None
    ):
        """
        A client to communicate with deepset Cloud evaluation sets and labels.

        :param client: deepset Cloud client
        :param workspace: Specifies the name of the workspace for which you want to create the client.
        :param evaluation_set: name of the evaluation set to fall back to

        """
        self.client = client
        self.workspace = workspace
        self.evaluation_set = evaluation_set

    def get_labels(self, evaluation_set: Optional[str], workspace: Optional[str] = None) -> List[Label]:
        """
        Searches for labels for a given evaluation set in deepset cloud. Returns a list of all found labels.
        If no labels were found, raises DeepsetCloudError.

        :param evaluation_set: name of the evaluation set for which labels should be fetched
        :param workspace: Specifies the name of the workspace on deepset Cloud.
                          If set to None, the EvaluationSetClient's default workspace (self.workspace) is used.

        :return: list of Label
        """
        url = f"{self._build_workspace_url(workspace=workspace)}/evaluation_sets/{evaluation_set}"
        response = self.client.get(url=url, raise_on_error=False)
        if response.status_code >= 400:
            raise DeepsetCloudError(f"No evaluation set found with the name {evaluation_set}")

        labels = response.json()

        return [
            Label(
                query=label_dict["query"],
                document=Document(content=label_dict["context"]),
                is_correct_answer=True,
                is_correct_document=True,
                origin="user-feedback",
                answer=Answer(label_dict["answer"]),
                id=label_dict["label_id"],
                pipeline_id=None,
                created_at=None,
                updated_at=None,
                meta=label_dict["meta"],
                filters={},
            )
            for label_dict in labels
        ]

    def get_labels_count(self, evaluation_set: Optional[str] = None, workspace: Optional[str] = None) -> int:
        """
        Counts labels for a given evaluation set in deepset cloud.

        :param evaluation_set: Optional evaluation set in deepset Cloud
                               If set to None, the EvaluationSetClient's default evaluation set (self.evaluation_set) is used.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
                          If set to None, the EvaluationSetClient's default workspace (self.workspace) is used.

        :return: Number of labels for the given (or defaulting) index
        """
        if not evaluation_set:
            evaluation_set = self.evaluation_set

        evaluation_set_response = self.get_evaluation_set(evaluation_set=evaluation_set, workspace=workspace)
        if evaluation_set_response is None:
            raise DeepsetCloudError(f"No evaluation set found with the name {evaluation_set}")

        return evaluation_set_response["total_labels"]

    def get_evaluation_sets(self, workspace: Optional[str] = None) -> List[dict]:
        """
        Searches for all evaluation set names in the given workspace in deepset Cloud.

        :param workspace: Specifies the name of the workspace on deepset Cloud.
                          If set to None, the EvaluationSetClient's default workspace (self.workspace) is used.

        :return: List of dictionaries that represent deepset Cloud evaluation sets.
                 These contain ("name", "evaluation_set_id", "created_at", "matched_labels", "total_labels") as fields.
        """
        evaluation_sets_response = self._get_evaluation_sets(workspace=workspace)

        return [eval_set for eval_set in evaluation_sets_response]

    def _get_evaluation_sets(self, workspace: Optional[str] = None) -> Generator:
        url = self._build_workspace_url(workspace=workspace)
        evaluation_set_url = f"{url}/evaluation_sets"
        return self.client.get_with_auto_paging(url=evaluation_set_url)

    def upload_evaluation_set(self, file_path: Path, workspace: Optional[str] = None):
        """
        Uploads an evaluation set.
        The name of file that you uploaded becomes the name of the evaluation set in deepset Cloud.
        When using Haystack annotation tool make sure to choose CSV as export format. The resulting file matches the expected format.

        Currently, deepset Cloud only supports CSV files (having "," as delimiter) with the following columns:
        - question (or query): the labelled question or query (required)
        - text: the answer to the question or relevant text to the query (required)
        - context: the surrounding words of the text (should be more than 100 characters) (optional)
        - file_name: the name of the file within the workspace that contains the text (optional)
        - answer_start: the character position within the file that marks the start of the text (optional)
        - answer_end: the character position within the file that marks the end of the text (optional)

        :param file_path: Path to the evaluation set file to be uploaded.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
                          If set to None, the EvaluationSetClient's default workspace (self.workspace) is used.
        """
        workspace_url = self._build_workspace_url(workspace)
        target_url = f"{workspace_url}/evaluation_sets/import"
        try:
            mime_type = guess_type(str(file_path))[0]
            with open(file_path, "rb") as file:
                self.client.post(url=target_url, files={"file": (file_path.name, file, mime_type)})
            logger.info(
                f"Successfully uploaded evaluation set file {file_path}. You can access it now under evaluation set '{file_path.name}'."
            )
        except DeepsetCloudError as e:
            logger.error("Error uploading evaluation set file %s: %s", file_path, e.args)

    def get_evaluation_set(
        self, evaluation_set: Optional[str] = None, workspace: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Returns information about the evaluation set.

        :param evaluation_set: Name of the evaluation set in deepset Cloud.
                               If set to None, the EvaluationSetClient's default evaluation set (self.evaluation_set) is used.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
                          If set to None, the EvaluationSetClient's default workspace (self.workspace) is used.

        :return: Dictionary that represents deepset Cloud evaluation sets.
                 These contain ("name", "evaluation_set_id", "created_at", "matched_labels", "total_labels") as fields.
        """
        url = self._build_workspace_url(workspace=workspace)
        evaluation_set_url = f"{url}/evaluation_sets"

        # evaluation_sets resource uses ids instead of names,
        # so we have to query by name (which works as a contains filter) and take the first entry with matching name
        query_params = {}
        if evaluation_set is not None:
            query_params["name"] = evaluation_set

        matches = [
            entry
            for entry in self.client.get_with_auto_paging(url=evaluation_set_url, query_params=query_params)
            if entry["name"] == evaluation_set
        ]
        if any(matches):
            return matches[0]
        return None

    def _build_workspace_url(self, workspace: Optional[str] = None):
        if workspace is None:
            workspace = self.workspace
        return self.client.build_workspace_url(workspace)


class FileClient:
    def __init__(self, client: DeepsetCloudClient, workspace: Optional[str] = None):
        """
        A client to manage files on deepset Cloud.

        :param client: deepset Cloud client
        :param workspace: Specifies the name of the workspace for which you want to create the client.
        """
        self.client = client
        self.workspace = workspace

    def upload_files(
        self,
        file_paths: List[Path],
        metas: Optional[List[Dict]] = None,
        workspace: Optional[str] = None,
        headers: Optional[dict] = None,
    ):
        """
        Uploads files to the deepset Cloud workspace.

        :param file_paths: File paths to upload (for example .txt or .pdf files)
        :param metas: Metadata of the files to upload
        :param workspace: Specifies the name of the workspace on deepset Cloud.
                          If set to None, the FileClient's default workspace is used.
        :param headers: Headers to pass to the API call
        """
        workspace_url = self._build_workspace_url(workspace)
        files_url = f"{workspace_url}/files"
        if metas is None:
            metas = [{} for _ in file_paths]

        file_ids = []
        for file_path, meta in tqdm(zip(file_paths, metas), total=len(file_paths)):
            try:
                mime_type = guess_type(str(file_path))[0]
                with open(file_path, "rb") as file:
                    response_file_upload = self.client.post(
                        url=files_url,
                        files={"file": (file_path.name, file, mime_type)},
                        data={"meta": json.dumps(meta)},
                        headers=headers,
                    )
                file_id = response_file_upload.json().get("file_id")
                file_ids.append(file_id)
            except Exception as e:
                logger.exception("Error uploading file %s", file_path)

        logger.info("Successfully uploaded %s files.", len(file_ids))

    def delete_file(self, file_id: str, workspace: Optional[str] = None, headers: Optional[dict] = None):
        """
        Delete a file from the deepset Cloud workspace.

        :param file_id: The id of the file to be deleted. Use `list_files` to retrieve the id of a file.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
                          If set to None, the FileClient's default workspace is used.
        :param headers: Headers to pass to the API call
        """
        workspace_url = self._build_workspace_url(workspace)
        file_url = f"{workspace_url}/files/{file_id}"
        self.client.delete(url=file_url, headers=headers)

    def delete_all_files(self, workspace: Optional[str] = None, headers: Optional[dict] = None):
        """
        Delete all files from a deepset Cloud workspace.

        :param workspace: Specifies the name of the workspace on deepset Cloud.
                          If set to None, the FileClient's default workspace is used.
        :param headers: Headers to pass to the API call.
        """
        workspace_url = self._build_workspace_url(workspace)
        file_url = f"{workspace_url}/files"
        self.client.delete(url=file_url, headers=headers)

    def list_files(
        self,
        name: Optional[str] = None,
        meta_key: Optional[str] = None,
        meta_value: Optional[str] = None,
        workspace: Optional[str] = None,
        headers: Optional[dict] = None,
    ) -> Generator:
        """
        List all files in the given deepset Cloud workspace.
        You can filter by name or by meta values.

        :param name: The name or part of the name of the file.
        :param meta_key: The key of the metadata of the file to be filtered for.
        :param meta_value: The value of the metadata of the file to be filtered for.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
                          If set to None, the FileClient's default workspace is used.
        :param headers: Headers to pass to the API call
        """
        workspace_url = self._build_workspace_url(workspace)
        files_url = f"{workspace_url}/files"
        query_params = {"name": name, "meta_key": meta_key, "meta_value": meta_value}
        generator = self.client.get_with_auto_paging(url=files_url, headers=headers, query_params=query_params)
        return generator

    def _build_workspace_url(self, workspace: Optional[str] = None):
        if workspace is None:
            workspace = self.workspace
        return self.client.build_workspace_url(workspace)


class EvaluationRunClient:
    def __init__(self, client: DeepsetCloudClient, workspace: Optional[str] = None):
        """
        A client to manage deepset Cloud evaluation runs.

        :param client: deepset Cloud client
        :param workspace: Specifies the name of the workspace for which you want to create the client.
        """
        self.client = client
        self.workspace = workspace

    def create_eval_run(
        self,
        eval_run_name: str,
        workspace: Optional[str] = None,
        pipeline_config_name: Optional[str] = None,
        headers: Optional[dict] = None,
        evaluation_set: Optional[str] = None,
        eval_mode: Literal["integrated", "isolated"] = "integrated",
        debug: bool = False,
        comment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Creates an evaluation run.

        :param eval_run_name: The name of the evaluation run.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
                          If set to None, the EvaluationRunClient's default workspace is used.
        :param pipeline_config_name: The name of the pipeline to evaluate.
        :param evaluation_set: The name of the evaluation set to use.
        :param eval_mode: The evaluation mode to use.
        :param debug: Wheter to enable debug output.
        :param comment: Comment to add about to the evaluation run.
        :param tags: Tags to add to the evaluation run.
        :param headers: Headers to pass to the API call
        """
        workspace_url = self._build_workspace_url(workspace)
        eval_run_url = f"{workspace_url}/eval_runs"
        response = self.client.post(
            eval_run_url,
            json={
                "pipeline_name": pipeline_config_name,
                "evaluation_set_name": evaluation_set,
                "debug": debug,
                "eval_mode": 0 if eval_mode == "integrated" else 1,
                "comment": comment,
                "name": eval_run_name,
                "tags": tags,
            },
            headers=headers,
        )
        return response.json()["data"]

    def get_eval_run(
        self, eval_run_name: str, workspace: Optional[str] = None, headers: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Gets the evaluation run and shows its parameters and metrics.

        :param eval_run_name: The name of the evaluation run.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
                          If set to None, the EvaluationRunClient's default workspace is used.
        :param headers: Headers to pass to the API call
        """
        workspace_url = self._build_workspace_url(workspace)
        eval_run_url = f"{workspace_url}/eval_runs/{eval_run_name}"
        response = self.client.get(eval_run_url, headers=headers)
        return response.json()

    def get_eval_runs(self, workspace: Optional[str] = None, headers: Optional[dict] = None) -> List[Dict[str, Any]]:
        """
        Gets all evaluation runs and shows its parameters and metrics.

        :param workspace: Specifies the name of the workspace on deepset Cloud.
                          If set to None, the EvaluationRunClient's default workspace is used.
        :param headers: Headers to pass to the API call
        """
        workspace_url = self._build_workspace_url(workspace)
        eval_run_url = f"{workspace_url}/eval_runs"
        response = self.client.get_with_auto_paging(eval_run_url, headers=headers)
        return [eval_run for eval_run in response]

    def delete_eval_run(self, eval_run_name: str, workspace: Optional[str] = None, headers: Optional[dict] = None):
        """
        Deletes an evaluation run.

        :param eval_run_name: The name of the evaluation run.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
                          If set to None, the EvaluationRunClient's default workspace is used.
        :param headers: Headers to pass to the API call
        """
        workspace_url = self._build_workspace_url(workspace)
        eval_run_url = f"{workspace_url}/eval_runs/{eval_run_name}"
        response = self.client.delete(eval_run_url, headers=headers)
        if response.status_code == 204:
            logger.info("Evaluation run '%s' deleted.", eval_run_name)

    def start_eval_run(self, eval_run_name: str, workspace: Optional[str] = None, headers: Optional[dict] = None):
        """
        Starts an evaluation run.

        :param eval_run_name: The name of the evaluation run.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
                          If set to None, the EvaluationRunClient's default workspace is used.
        :param headers: Headers to pass to the API call
        """
        workspace_url = self._build_workspace_url(workspace)
        eval_run_url = f"{workspace_url}/eval_runs/{eval_run_name}/start"
        response = self.client.post(eval_run_url, headers=headers)
        if response.status_code == 204:
            logger.info("Evaluation run '%s' has been started.", eval_run_name)

    def update_eval_run(
        self,
        eval_run_name: str,
        workspace: Optional[str] = None,
        pipeline_config_name: Optional[str] = None,
        headers: Optional[dict] = None,
        evaluation_set: Optional[str] = None,
        eval_mode: Literal["integrated", "isolated", None] = None,
        debug: Optional[bool] = None,
        comment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Updates an evaluation run.

        :param eval_run_name: The name of the evaluation run to update.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
                          If set to None, the FileClient's default workspace is used.
        :param pipeline_config_name: The name of the pipeline to evaluate.
        :param evaluation_set: The name of the evaluation set to use.
        :param eval_mode: The evaluation mode to use.
        :param debug: Wheter to enable debug output.
        :param comment: Comment to add about to the evaluation run.
        :param tags: Tags to add to the evaluation run.
        :param headers: Headers to pass to the API call
        """
        workspace_url = self._build_workspace_url(workspace)
        eval_run_url = f"{workspace_url}/eval_runs/{eval_run_name}"
        eval_mode_param = None
        if eval_mode is not None:
            eval_mode_param = 0 if eval_mode == "integrated" else 1
        response = self.client.patch(
            eval_run_url,
            json={
                "pipeline_name": pipeline_config_name,
                "evaluation_set_name": evaluation_set,
                "debug": debug,
                "eval_mode": eval_mode_param,
                "comment": comment,
                "tags": tags,
            },
            headers=headers,
        )
        return response.json()["data"]

    def get_eval_run_results(
        self, eval_run_name: str, workspace: Optional[str] = None, headers: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Collects and returns the predictions of an evaluation run.

        :param eval_run_name: The name of the evaluation run to fetch results for.
        :param workspace: Specifies the name of the deepset Cloud workspace where the evaluation run exists.
                          If set to None, the EvaluationRunClient's default workspace is used.
        :param headers: The headers that you want to pass to the API call.
        """

        response = self.get_eval_run(eval_run_name, workspace, headers)
        predictions_per_node = {}
        for eval_result in response["eval_results"]:
            predictions_per_node[eval_result["node_name"]] = self.get_eval_run_predictions(
                eval_run_name=eval_run_name, node_name=eval_result["node_name"], workspace=workspace, headers=headers
            )

        return predictions_per_node

    def get_eval_run_predictions(
        self, eval_run_name: str, node_name: str, workspace: Optional[str] = None, headers: Optional[dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetches predictions for the evaluation run and a node name you specify.

        :param eval_run_name: The name of the evaluation run to fetch predictions for.
        :param node_name: The name of the node to fetch predictions for.
        :param workspace: Specifies the name of the deepset Cloud workspace where the evaluation run exists.
                          If set to None, the EvaluationRunClient's default workspace is used.
        :param headers: The headers that you want to pass to the API call.
        """
        workspace_url = self._build_workspace_url(workspace)
        eval_run_prediction_url = f"{workspace_url}/eval_runs/{eval_run_name}/nodes/{node_name}/predictions"
        response = self.client.get_with_auto_paging(eval_run_prediction_url, headers=headers)
        return [prediction for prediction in response]

    def _build_workspace_url(self, workspace: Optional[str] = None):
        if workspace is None:
            workspace = self.workspace
        return self.client.build_workspace_url(workspace)


class DeepsetCloud:
    """
    A facade to communicate with deepset Cloud.
    """

    @classmethod
    def get_index_client(
        cls,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        workspace: str = "default",
        index: Optional[str] = None,
    ) -> IndexClient:
        """
        Creates a client to communicate with deepset Cloud indexes.

        :param api_key: Secret value of the API key.
                        If not specified, it's read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the deepset Cloud API.
                             If not specified, it's read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
                             If environment variable is not set, defaults to 'https://api.cloud.deepset.ai/api/v1'.
        :param workspace: Specifies the name of the workspace for which you want to create the client.
        :param index: index in deepset Cloud workspace

        """
        client = DeepsetCloudClient(api_key=api_key, api_endpoint=api_endpoint)
        return IndexClient(client=client, workspace=workspace, index=index)

    @classmethod
    def get_pipeline_client(
        cls,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        workspace: str = "default",
        pipeline_config_name: Optional[str] = None,
    ) -> PipelineClient:
        """
        Creates a client to communicate with deepset Cloud pipelines.

        :param api_key: Secret value of the API key.
                        If not specified, it's read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the deepset Cloud API.
                             If not specified, it's read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
                             If environment variable is not set, defaults to 'https://api.cloud.deepset.ai/api/v1'.
        :param workspace: Specifies the name of the workspace for which you want to create the client.
        :param pipeline_config_name: name of the pipeline_config in deepset Cloud workspace

        """
        client = DeepsetCloudClient(api_key=api_key, api_endpoint=api_endpoint)
        return PipelineClient(client=client, workspace=workspace, pipeline_config_name=pipeline_config_name)

    @classmethod
    def get_evaluation_set_client(
        cls,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        workspace: str = "default",
        evaluation_set: str = "default",
    ) -> EvaluationSetClient:
        """
        Creates a client to communicate with deepset Cloud labels.

        :param api_key: Secret value of the API key.
                        If not specified, it's read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the deepset Cloud API.
                             If not specified, it's read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
                             If environment variable is not set, defaults to 'https://api.cloud.deepset.ai/api/v1'.
        :param workspace: Specifies the name of the workspace for which you want to create the client.
        :param evaluation_set: name of the evaluation set in deepset Cloud

        """
        client = DeepsetCloudClient(api_key=api_key, api_endpoint=api_endpoint)
        return EvaluationSetClient(client=client, workspace=workspace, evaluation_set=evaluation_set)

    @classmethod
    def get_eval_run_client(
        cls, api_key: Optional[str] = None, api_endpoint: Optional[str] = None, workspace: str = "default"
    ) -> EvaluationRunClient:
        """
        Creates a client to manage evaluation runs on deepset Cloud.

        :param api_key: Secret value of the API key.
                        If not specified, it's read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the deepset Cloud API.
                             If not specified, it's read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
                             If environment variable is not set, defaults to 'https://api.cloud.deepset.ai/api/v1'.
        :param workspace: Specifies the name of the workspace for which you want to create the client.

        """
        client = DeepsetCloudClient(api_key=api_key, api_endpoint=api_endpoint)
        return EvaluationRunClient(client=client, workspace=workspace)

    @classmethod
    def get_file_client(
        cls, api_key: Optional[str] = None, api_endpoint: Optional[str] = None, workspace: str = "default"
    ) -> FileClient:
        """
        Creates a client to manage files on deepset Cloud.

        :param api_key: Secret value of the API key.
                        If not specified, it's read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the deepset Cloud API.
                             If not specified, it's read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
                             If environment variable is not set, defaults to 'https://api.cloud.deepset.ai/api/v1'.
        :param workspace: Specifies the name of the workspace for which you want to create the client.

        """
        client = DeepsetCloudClient(api_key=api_key, api_endpoint=api_endpoint)
        return FileClient(client=client, workspace=workspace)


class DeepsetCloudExperiments:
    """
    A facade to conduct and manage experiments within deepset Cloud.

    To start a new experiment run:
    1. Choose a pipeline to evaluate using `list_pipelines()`.
    2. Choose an evaluation set using `list_evaluation_sets()`.
    3. Create and start a new run using `create_and_start_run()`.
    4. Track the run using `get_run()`. When the run finishes, you can use the `eval_results` key in the returned dictionary to view the metrics.
    5. Inspect the result of a run in detail using `get_run_result()`.
       This returns an `EvaluationResult` object containing all the predictions and gold labels in the form of pandas dataframes.
       Use `calculate_metrics()` to recalculate metrics using different settings (for example, `top_k`) and `wrong_examples()` to show worst performing queries/labels.
    """

    @classmethod
    def list_pipelines(
        cls, workspace: str = "default", api_key: Optional[str] = None, api_endpoint: Optional[str] = None
    ) -> List[dict]:
        """
        Lists all pipelines available on deepset Cloud.

        :param workspace: Specifies the name of the workspace on deepset Cloud.
        :param api_key: Secret value of the API key.
                        If not specified, it's read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the deepset Cloud API.
                             If not specified, it's read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
                             If environment variable is not set, defaults to 'https://api.cloud.deepset.ai/api/v1'.

        Returns:
            list of dictionaries: List[dict]
            each dictionary: {
                        "name": str -> `pipeline_config_name` to be used in `load_from_deepset_cloud()`,
                        "..." -> additional pipeline meta information
                        }
            example:

            ```python
            [{'name': 'my_super_nice_pipeline_config',
                'pipeline_id': '2184e0c1-c6ec-40a1-9b28-5d2768e5efa2',
                'status': 'DEPLOYED',
                'created_at': '2022-02-01T09:57:03.803991+00:00',
                'deleted': False,
                'is_default': False,
                'indexing': {'status': 'IN_PROGRESS',
                'pending_file_count': 3,
                'total_file_count': 31}}]
            ```

        """
        client = DeepsetCloud.get_pipeline_client(api_key=api_key, api_endpoint=api_endpoint, workspace=workspace)
        pipeline_config_infos = list(client.list_pipeline_configs())
        return pipeline_config_infos

    @classmethod
    def list_evaluation_sets(
        cls, workspace: str = "default", api_key: Optional[str] = None, api_endpoint: Optional[str] = None
    ) -> List[dict]:
        """
        Lists all evaluation sets available on deepset Cloud.

        :param workspace: Specifies the name of the workspace on deepset Cloud.
        :param api_key: Secret value of the API key.
                        If not specified, it's read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the deepset Cloud API.
                             If not specified, it's read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
                             If environment variable is not set, defaults to 'https://api.cloud.deepset.ai/api/v1'.

        Returns:
            list of dictionaries: List[dict]
            each dictionary: {
                        "name": str -> `evaluation_set` to be used in `create_run()`,
                        "..." -> additional pipeline meta information
                        }
            example:

            ```python
            [{'evaluation_set_id': 'fb084729-57ad-4b57-9f78-ec0eb4d29c9f',
                'name': 'my-question-answering-evaluation-set',
                'created_at': '2022-05-06T09:54:14.830529+00:00',
                'matched_labels': 234,
                'total_labels': 234}]
            ```
        """
        client = DeepsetCloud.get_evaluation_set_client(api_key=api_key, api_endpoint=api_endpoint, workspace=workspace)
        return client.get_evaluation_sets()

    @classmethod
    def get_runs(
        cls, workspace: str = "default", api_key: Optional[str] = None, api_endpoint: Optional[str] = None
    ) -> List[dict]:
        """
        Gets all evaluation runs.

        :param workspace: Specifies the name of the workspace on deepset Cloud.
        :param api_key: Secret value of the API key.
                        If not specified, it's read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the deepset Cloud API.
                             If not specified, it's read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
                             If environment variable is not set, defaults to 'https://api.cloud.deepset.ai/api/v1'.

        Returns:
            list of dictionaries: List[dict]
            example:

            ```python
            [{'eval_run_name': 'my-eval-run-1',
                'parameters': {
                    'pipeline_name': 'my-pipeline-1_696bc5d0-ee65-46c1-a308-059507bc353b',
                    'evaluation_set_name': 'my-eval-set-name',
                    'debug': False,
                    'eval_mode': 0
                },
                'metrics': {
                    'isolated_exact_match': 0.45,
                    'isolated_f1': 0.89,
                    'isolated_sas': 0.91,
                    'integrated_exact_match': 0.39,
                    'integrated_f1': 0.76,
                    'integrated_sas': 0.78,
                    'mean_reciprocal_rank': 0.77,
                    'mean_average_precision': 0.78,
                    'recall_single_hit': 0.91,
                    'recall_multi_hit': 0.91,
                    'normal_discounted_cummulative_gain': 0.83,
                    'precision': 0.52
                },
                'logs': {},
                'status': 1,
                'eval_mode': 0,
                'eval_run_labels': [],
                'created_at': '2022-05-24T12:13:16.445857+00:00',
                'comment': 'This is a comment about thiseval run',
                'tags': ['experiment-1', 'experiment-2', 'experiment-3']
                }]
            ```
        """
        client = DeepsetCloud.get_eval_run_client(api_key=api_key, api_endpoint=api_endpoint, workspace=workspace)
        return client.get_eval_runs()

    @classmethod
    def create_run(
        cls,
        eval_run_name: str,
        workspace: str = "default",
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        pipeline_config_name: Optional[str] = None,
        evaluation_set: Optional[str] = None,
        eval_mode: Literal["integrated", "isolated"] = "integrated",
        debug: bool = False,
        comment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Creates an evaluation run.

        :param eval_run_name: The name of the evaluation run.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
                          If set to None, the EvaluationRunClient's default workspace is used.
        :param pipeline_config_name: The name of the pipeline to evaluate. Use `list_pipelines()` to list all available pipelines.
        :param evaluation_set: The name of the evaluation set to use. Use `list_evaluation_sets()` to list all available evaluation sets.
        :param eval_mode: The evaluation mode to use.
        :param debug: Wheter to enable debug output.
        :param comment: Comment to add about to the evaluation run.
        :param tags: Tags to add to the evaluation run.
        :param api_key: Secret value of the API key.
                        If not specified, it's read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the deepset Cloud API.
                             If not specified, it's read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
                             If environment variable is not set, defaults to 'https://api.cloud.deepset.ai/api/v1'.
        """
        client = DeepsetCloud.get_eval_run_client(api_key=api_key, api_endpoint=api_endpoint, workspace=workspace)
        return client.create_eval_run(
            eval_run_name=eval_run_name,
            pipeline_config_name=pipeline_config_name,
            evaluation_set=evaluation_set,
            eval_mode=eval_mode,
            debug=debug,
            comment=comment,
            tags=tags,
        )

    @classmethod
    def update_run(
        cls,
        eval_run_name: str,
        workspace: str = "default",
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        pipeline_config_name: Optional[str] = None,
        evaluation_set: Optional[str] = None,
        eval_mode: Literal["integrated", "isolated"] = "integrated",
        debug: bool = False,
        comment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Updates an evaluation run.

        :param eval_run_name: The name of the evaluation run to update.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
                          If set to None, the FileClient's default workspace is used.
        :param pipeline_config_name: The name of the pipeline to evaluate. Use `list_pipelines()` to list all available pipelines.
        :param evaluation_set: The name of the evaluation set to use. Use `list_evaluation_sets()` to list all available evaluation sets.
        :param eval_mode: The evaluation mode to use.
        :param debug: Wheter to enable debug output.
        :param comment: Comment to add about to the evaluation run.
        :param tags: Tags to add to the evaluation run.
        :param api_key: Secret value of the API key.
                        If not specified, it's read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the deepset Cloud API.
                             If not specified, it's read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
                             If environment variable is not set, defaults to 'https://api.cloud.deepset.ai/api/v1'.
        """
        client = DeepsetCloud.get_eval_run_client(api_key=api_key, api_endpoint=api_endpoint, workspace=workspace)
        return client.update_eval_run(
            eval_run_name=eval_run_name,
            pipeline_config_name=pipeline_config_name,
            evaluation_set=evaluation_set,
            eval_mode=eval_mode,
            debug=debug,
            comment=comment,
            tags=tags,
        )

    @classmethod
    def get_run(
        cls,
        eval_run_name: str,
        workspace: str = "default",
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Gets the evaluation run and shows its parameters and metrics.

        :param eval_run_name: The name of the evaluation run.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
                          If set to None, the EvaluationRunClient's default workspace is used.
        :param api_key: Secret value of the API key.
                        If not specified, it's read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the deepset Cloud API.
                             If not specified, it's read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
                             If environment variable is not set, defaults to 'https://api.cloud.deepset.ai/api/v1'.
        """
        client = DeepsetCloud.get_eval_run_client(api_key=api_key, api_endpoint=api_endpoint, workspace=workspace)
        return client.get_eval_run(eval_run_name=eval_run_name)

    @classmethod
    def delete_run(
        cls,
        eval_run_name: str,
        workspace: str = "default",
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
    ):
        """
        Deletes an evaluation run.

        :param eval_run_name: The name of the evaluation run.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
                          If set to None, the EvaluationRunClient's default workspace is used.
        :param api_key: Secret value of the API key.
                        If not specified, it's read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the deepset Cloud API.
                             If not specified, it's read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
                             If environment variable is not set, defaults to 'https://api.cloud.deepset.ai/api/v1'.
        """
        client = DeepsetCloud.get_eval_run_client(api_key=api_key, api_endpoint=api_endpoint, workspace=workspace)
        return client.delete_eval_run(eval_run_name=eval_run_name)

    @classmethod
    def start_run(
        cls,
        eval_run_name: str,
        workspace: str = "default",
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
    ):
        """
        Starts an evaluation run.

        :param eval_run_name: The name of the evaluation run.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
                          If set to None, the EvaluationRunClient's default workspace is used.
        :param api_key: Secret value of the API key.
                        If not specified, it's read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the deepset Cloud API.
                             If not specified, it's read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
                             If environment variable is not set, defaults to 'https://api.cloud.deepset.ai/api/v1'.
        """
        client = DeepsetCloud.get_eval_run_client(api_key=api_key, api_endpoint=api_endpoint, workspace=workspace)
        client.start_eval_run(eval_run_name=eval_run_name)
        logger.info("You can check run progess by inspecting the `status` field returned from `get_run()`.")

    @classmethod
    def create_and_start_run(
        cls,
        eval_run_name: str,
        workspace: str = "default",
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        pipeline_config_name: Optional[str] = None,
        evaluation_set: Optional[str] = None,
        eval_mode: Literal["integrated", "isolated"] = "integrated",
        debug: bool = False,
        comment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Creates and starts an evaluation run.

        :param eval_run_name: The name of the evaluation run.
        :param workspace: Specifies the name of the workspace on deepset Cloud.
                          If set to None, the EvaluationRunClient's default workspace is used.
        :param pipeline_config_name: The name of the pipeline to evaluate. Use `list_pipelines()` to list all available pipelines.
        :param evaluation_set: The name of the evaluation set to use. Use `list_evaluation_sets()` to list all available evaluation sets.
        :param eval_mode: The evaluation mode to use.
        :param debug: Wheter to enable debug output.
        :param comment: Comment to add about to the evaluation run.
        :param tags: Tags to add to the evaluation run.
        :param api_key: Secret value of the API key.
                        If not specified, it's read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the deepset Cloud API.
                             If not specified, it's read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
                             If environment variable is not set, defaults to 'https://api.cloud.deepset.ai/api/v1'.
        """
        cls.create_run(
            eval_run_name=eval_run_name,
            workspace=workspace,
            api_key=api_key,
            api_endpoint=api_endpoint,
            pipeline_config_name=pipeline_config_name,
            evaluation_set=evaluation_set,
            eval_mode=eval_mode,
            debug=debug,
            comment=comment,
            tags=tags,
        )
        cls.start_run(eval_run_name=eval_run_name, workspace=workspace, api_key=api_key, api_endpoint=api_endpoint)

    @classmethod
    def get_run_result(
        cls,
        eval_run_name: str,
        workspace: str = "default",
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Fetches the results of an evaluation run and turns them into an EvaluationResult object.

        :param eval_run_name: The name of the evaluation run whose results you want to fetch.
        :param workspace: Specifies the name of the deepset Cloud workspace where the evaluation run exists.
                          If set to None, the EvaluationRunClient's default workspace is used.
        :param api_key: Secret value of the API key.
                        If not specified, it's read from the DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the deepset Cloud API.
                             If not specified, it's read from the DEEPSET_CLOUD_API_ENDPOINT environment variable.
                             If the environment variable is not set, it defaults to 'https://api.cloud.deepset.ai/api/v1'.
        """
        client = DeepsetCloud.get_eval_run_client(api_key=api_key, api_endpoint=api_endpoint, workspace=workspace)
        results = client.get_eval_run_results(eval_run_name=eval_run_name, workspace=workspace)

        # cast node results in-memory from json to pandas dataframe
        results = {node_name: pd.DataFrame(node_predictions) for node_name, node_predictions in results.items()}

        return EvaluationResult(results)
