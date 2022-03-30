from __future__ import annotations
from enum import Enum
import logging
import os
import time
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from haystack.schema import Label, Document, Answer

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import requests
import yaml

DEFAULT_API_ENDPOINT = f"DC_API_PLACEHOLDER/v1"  # TODO


class PipelineStatus(Enum):
    UNDEPLOYED: str = "UNDEPLOYED"
    DEPLOYED_UNHEALTHY: str = "DEPLOYED_UNHEALTHY"
    DEPLOYED: str = "DEPLOYED"
    DEPLOYMENT_IN_PROGRESS: str = "DEPLOYMENT_IN_PROGRESS"
    UNDEPLOYMENT_IN_PROGRESS: str = "UNDEPLOYMENT_IN_PROGRESS"
    DEPLOYMENT_SCHEDULED: str = "DEPLOYMENT_SCHEDULED"
    UNDEPLOYMENT_SCHEDULED: str = "UNDEPLOYMENT_SCHEDULED"
    UKNOWN: str = "UNKNOWN"

    @classmethod
    def from_str(cls, status_string: str) -> PipelineStatus:
        return cls.__dict__.get(status_string, PipelineStatus.UKNOWN)


SATISFIED_STATES_KEY = "satisfied_states"
VALID_INITIAL_STATES_KEY = "valid_initial_states"
VALID_TRANSITIONING_STATES_KEY = "valid_transitioning_states"
PIPELINE_STATE_TRANSITION_INFOS: Dict[PipelineStatus, Dict[str, List[PipelineStatus]]] = {
    PipelineStatus.UNDEPLOYED: {
        SATISFIED_STATES_KEY: [PipelineStatus.UNDEPLOYED],
        VALID_INITIAL_STATES_KEY: [PipelineStatus.DEPLOYED, PipelineStatus.DEPLOYED_UNHEALTHY],
        VALID_TRANSITIONING_STATES_KEY: [
            PipelineStatus.UNDEPLOYMENT_SCHEDULED,
            PipelineStatus.UNDEPLOYMENT_IN_PROGRESS,
        ],
    },
    PipelineStatus.DEPLOYED: {
        SATISFIED_STATES_KEY: [PipelineStatus.DEPLOYED, PipelineStatus.DEPLOYED_UNHEALTHY],
        VALID_INITIAL_STATES_KEY: [PipelineStatus.UNDEPLOYED],
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
    """Raised when there is an error communicating with Deepset Cloud"""


class DeepsetCloudClient:
    def __init__(self, api_key: str = None, api_endpoint: Optional[str] = None):
        """
        A client to communicate with Deepset Cloud.

        :param api_key: Secret value of the API key.
                        If not specified, will be read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the Deepset Cloud API.
                             If not specified, will be read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
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
        query_params: dict = None,
        headers: dict = None,
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
        query_params: dict = None,
        headers: dict = None,
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
        data: Any = None,
        query_params: dict = None,
        headers: dict = None,
        stream: bool = False,
        raise_on_error: bool = True,
    ):
        return self._execute_request(
            method="POST",
            url=url,
            query_params=query_params,
            json=json,
            data=data,
            stream=stream,
            headers=headers,
            raise_on_error=raise_on_error,
        )

    def post_with_auto_paging(
        self,
        url: str,
        json: dict = {},
        data: Any = None,
        query_params: dict = None,
        headers: dict = None,
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
        json: dict = None,
        data: Any = None,
        query_params: dict = None,
        stream: bool = False,
        headers: dict = None,
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
        data: Any = None,
        query_params: dict = None,
        headers: dict = None,
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

    def _execute_auto_paging_request(
        self,
        method: Literal["GET", "POST", "PUT", "HEAD"],
        url: str,
        json: dict = None,
        data: Any = None,
        query_params: dict = None,
        headers: dict = None,
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
        method: Literal["GET", "POST", "PUT", "HEAD"],
        url: str,
        json: dict = None,
        data: Any = None,
        query_params: dict = None,
        headers: dict = None,
        stream: bool = False,
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
        )
        if raise_on_error and response.status_code > 299:
            raise DeepsetCloudError(
                f"{method} {url} failed: HTTP {response.status_code} - {response.reason}\n{response.content.decode()}"
            )
        return response

    def build_workspace_url(self, workspace: str = None):
        api_endpoint = f"{self.api_endpoint}".rstrip("/")
        url = f"{api_endpoint}/workspaces/{workspace}"
        return url

    def _remove_null_values(self, body: dict) -> dict:
        return {k: v for k, v in body.items() if v is not None}


class IndexClient:
    def __init__(self, client: DeepsetCloudClient, workspace: Optional[str] = None, index: Optional[str] = None):
        """
        A client to communicate with Deepset Cloud indexes.

        :param client: Deepset Cloud client
        :param workspace: workspace in Deepset Cloud
        :param index: index in Deepset Cloud workspace

        """
        self.client = client
        self.workspace = workspace
        self.index = index

    def info(self, workspace: Optional[str] = None, index: Optional[str] = None, headers: dict = None):
        index_url = self._build_index_url(workspace=workspace, index=index)
        try:
            response = self.client.get(url=index_url, headers=headers)
            return response.json()
        except Exception as ie:
            raise DeepsetCloudError(f"Could not connect to Deepset Cloud:\n{ie}") from ie

    def query(
        self,
        query: Optional[str] = None,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        top_k: int = 10,
        custom_query: Optional[str] = None,
        query_emb: Optional[List[float]] = None,
        return_embedding: Optional[bool] = None,
        similarity: Optional[str] = None,
        workspace: Optional[str] = None,
        index: Optional[str] = None,
        all_terms_must_match: Optional[bool] = None,
        headers: dict = None,
    ) -> List[dict]:
        index_url = self._build_index_url(workspace=workspace, index=index)
        query_url = f"{index_url}/documents-query"
        request = {
            "query": query,
            "filters": filters,
            "top_k": top_k,
            "custom_query": custom_query,
            "query_emb": query_emb,
            "similarity": similarity,
            "return_embedding": return_embedding,
            "all_terms_must_match": all_terms_must_match,
        }
        response = self.client.post(url=query_url, json=request, headers=headers)
        return response.json()

    def stream_documents(
        self,
        return_embedding: Optional[bool] = False,
        filters: Optional[dict] = None,
        workspace: Optional[str] = None,
        index: Optional[str] = None,
        headers: dict = None,
    ):
        index_url = self._build_index_url(workspace=workspace, index=index)
        query_url = f"{index_url}/documents-stream"
        request = {"return_embedding": return_embedding, "filters": filters}
        response = self.client.post(url=query_url, json=request, headers=headers, stream=True)
        return response.iter_lines()

    def get_document(
        self,
        id: str,
        return_embedding: Optional[bool] = False,
        workspace: Optional[str] = None,
        index: Optional[str] = None,
        headers: dict = None,
    ):
        index_url = self._build_index_url(workspace=workspace, index=index)
        document_url = f"{index_url}/documents/{id}"
        query_params = {"return_embedding": return_embedding}
        response = self.client.get(url=document_url, headers=headers, query_params=query_params, raise_on_error=False)
        doc: Optional[dict] = None
        if response.status_code == 200:
            doc = response.json()
        else:
            logger.warning(
                f"Document {id} could not be fetched from Deepset Cloud: HTTP {response.status_code} - {response.reason}\n{response.content.decode()}"
            )
        return doc

    def count_documents(
        self,
        filters: Optional[dict] = None,
        only_documents_without_embedding: Optional[bool] = False,
        workspace: Optional[str] = None,
        index: Optional[str] = None,
        headers: dict = None,
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
        A client to communicate with Deepset Cloud pipelines.

        :param client: Deepset Cloud client
        :param workspace: workspace in Deepset Cloud
        :param pipeline_config_name: name of the pipeline_config in Deepset Cloud workspace

        """
        self.client = client
        self.workspace = workspace
        self.pipeline_config_name = pipeline_config_name

    def get_pipeline_config(
        self, workspace: Optional[str] = None, pipeline_config_name: Optional[str] = None, headers: dict = None
    ) -> dict:
        pipeline_url = self._build_pipeline_url(workspace=workspace, pipeline_config_name=pipeline_config_name)
        pipeline_config_url = f"{pipeline_url}/json"
        response = self.client.get(url=pipeline_config_url, headers=headers).json()
        return response

    def get_pipeline_config_info(
        self, workspace: Optional[str] = None, pipeline_config_name: Optional[str] = None, headers: dict = None
    ) -> Optional[dict]:
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

    def list_pipeline_configs(self, workspace: Optional[str] = None, headers: dict = None) -> Generator:
        workspace_url = self._build_workspace_url(workspace)
        pipelines_url = f"{workspace_url}/pipelines"
        generator = self.client.get_with_auto_paging(url=pipelines_url, headers=headers)
        return generator

    def save_pipeline_config(
        self,
        config: dict,
        pipeline_config_name: Optional[str] = None,
        workspace: Optional[str] = None,
        headers: dict = None,
    ):
        config["name"] = pipeline_config_name
        workspace_url = self._build_workspace_url(workspace=workspace)
        pipelines_url = f"{workspace_url}/pipelines"
        response = self.client.post(url=pipelines_url, data=yaml.dump(config), headers=headers).json()
        if "name" not in response or response["name"] != pipeline_config_name:
            logger.warning(f"Unexpected response from saving pipeline config: {response}")

    def update_pipeline_config(
        self,
        config: dict,
        pipeline_config_name: Optional[str] = None,
        workspace: Optional[str] = None,
        headers: dict = None,
    ):
        config["name"] = pipeline_config_name
        pipeline_url = self._build_pipeline_url(workspace=workspace, pipeline_config_name=pipeline_config_name)
        yaml_url = f"{pipeline_url}/yaml"
        response = self.client.put(url=yaml_url, data=yaml.dump(config), headers=headers).json()
        if "name" not in response or response["name"] != pipeline_config_name:
            logger.warning(f"Unexpected response from updating pipeline config: {response}")

    def deploy(
        self, pipeline_config_name: Optional[str] = None, workspace: str = None, headers: dict = None, timeout: int = 60
    ):
        """
        Deploys the pipelines of a pipeline config on Deepset Cloud.
        Blocks until pipelines are successfully deployed, deployment failed or timeout exceeds.
        If pipelines are already deployed no action will be taken and an info will be logged.
        If timeout exceeds a TimeoutError will be raised.
        If deployment fails a DeepsetCloudError will be raised.

        :param pipeline_config_name: name of the config file inside the Deepset Cloud workspace.
        :param workspace: workspace in Deepset Cloud
        :param headers: Headers to pass to API call
        :param timeout: The time in seconds to wait until deployment completes.
                        If the timeout is exceeded an error will be raised.
        """
        status, changed = self._transition_pipeline_state(
            target_state=PipelineStatus.DEPLOYED,
            timeout=timeout,
            pipeline_config_name=pipeline_config_name,
            workspace=workspace,
            headers=headers,
        )

        if status == PipelineStatus.DEPLOYED:
            if changed:
                logger.info(f"Pipeline config '{pipeline_config_name}' successfully deployed.")
            else:
                logger.info(f"Pipeline config '{pipeline_config_name}' is already deployed.")
        elif status == PipelineStatus.DEPLOYED_UNHEALTHY:
            logger.warning(
                f"Deployment of pipeline config '{pipeline_config_name}' succeeded. But '{pipeline_config_name}' is unhealthy."
            )
        elif status in [PipelineStatus.UNDEPLOYMENT_IN_PROGRESS, PipelineStatus.UNDEPLOYMENT_SCHEDULED]:
            raise DeepsetCloudError(
                f"Deployment of pipline config '{pipeline_config_name}' aborted. Undeployment was requested."
            )
        elif status == PipelineStatus.UNDEPLOYED:
            raise DeepsetCloudError(f"Deployment of pipeline config '{pipeline_config_name}' failed.")
        else:
            raise DeepsetCloudError(
                f"Deployment of pipeline config '{pipeline_config_name} ended in unexpected status: {status.value}"
            )

    def undeploy(
        self, pipeline_config_name: Optional[str] = None, workspace: str = None, headers: dict = None, timeout: int = 60
    ):
        """
        Undeploys the pipelines of a pipeline config on Deepset Cloud.
        Blocks until pipelines are successfully undeployed, undeployment failed or timeout exceeds.
        If pipelines are already undeployed no action will be taken and an info will be logged.
        If timeout exceeds a TimeoutError will be raised.
        If deployment fails a DeepsetCloudError will be raised.

        :param pipeline_config_name: name of the config file inside the Deepset Cloud workspace.
        :param workspace: workspace in Deepset Cloud
        :param headers: Headers to pass to API call
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
                logger.info(f"Pipeline config '{pipeline_config_name}' successfully undeployed.")
            else:
                logger.info(f"Pipeline config '{pipeline_config_name}' is already undeployed.")
        elif status in [PipelineStatus.DEPLOYMENT_IN_PROGRESS, PipelineStatus.DEPLOYMENT_SCHEDULED]:
            raise DeepsetCloudError(
                f"Undeployment of pipline config '{pipeline_config_name}' aborted. Deployment was requested."
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
        workspace: str = None,
        headers: dict = None,
    ) -> Tuple[PipelineStatus, bool]:
        """
        Transitions the pipeline config state to desired target_state on Deepset Cloud.

        :param target_state: the target state of the Pipeline config.
        :param pipeline_config_name: name of the config file inside the Deepset Cloud workspace.
        :param workspace: workspace in Deepset Cloud
        :param headers: Headers to pass to API call
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
        valid_transitioning_states = transition_info[VALID_TRANSITIONING_STATES_KEY]
        valid_initial_states = transition_info[VALID_INITIAL_STATES_KEY]

        status = PipelineStatus.from_str(pipeline_info["status"])
        if status in satisfied_states:
            return status, False

        if status not in valid_initial_states:
            raise DeepsetCloudError(
                f"Pipeline config '{pipeline_config_name}' is in invalid state '{status.value}' to be transitioned to '{target_state.value}'."
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
                logger.info(f"Current status of '{pipeline_config_name}' is: '{status}'")
                time.sleep(5)

        return status, True

    def _deploy(
        self, pipeline_config_name: Optional[str] = None, workspace: Optional[str] = None, headers: dict = None
    ) -> dict:
        pipeline_url = self._build_pipeline_url(workspace=workspace, pipeline_config_name=pipeline_config_name)
        deploy_url = f"{pipeline_url}/deploy"
        response = self.client.post(url=deploy_url, headers=headers).json()
        return response

    def _undeploy(
        self, pipeline_config_name: Optional[str] = None, workspace: Optional[str] = None, headers: dict = None
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
        A client to communicate with Deepset Cloud evaluation sets and labels.

        :param client: Deepset Cloud client
        :param workspace: workspace in Deepset Cloud
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
        :param workspace: Optional workspace in Deepset Cloud
                          If None, the EvaluationSetClient's default workspace (self.workspace) will be used.

        :return: list of Label
        """
        try:
            evaluation_sets_response = next(
                self._get_evaluation_set(evaluation_set=evaluation_set, workspace=workspace)
            )
        except StopIteration:
            raise DeepsetCloudError(f"No evaluation set found with the name {evaluation_set}")

        labels = self._get_labels_from_evaluation_set(
            workspace=workspace, evaluation_set_id=evaluation_sets_response["evaluation_set_id"]
        )

        return [
            Label(
                query=label_dict["query"],
                document=Document(content=label_dict["context"]),
                is_correct_answer=True,
                is_correct_document=True,
                origin="user-feedback",
                answer=Answer(label_dict["answer"]),
                id=label_dict["label_id"],
                no_answer=False if label_dict.get("answer", None) else True,
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
                               If None, the EvaluationSetClient's default evaluation set (self.evaluation_set) will be used.
        :param workspace: Optional workspace in deepset Cloud
                          If None, the EvaluationSetClient's default workspace (self.workspace) will be used.

        :return: Number of labels for the given (or defaulting) index
        """
        try:
            evaluation_sets_response = next(
                self._get_evaluation_set(evaluation_set=evaluation_set, workspace=workspace)
            )
        except StopIteration:
            raise DeepsetCloudError(f"No evaluation set found with the name {evaluation_set}")

        return evaluation_sets_response["total_labels"]

    def get_evaluation_sets(self, workspace: Optional[str] = None) -> List[dict]:
        """
        Searches for all evaluation set names in the given workspace in Deepset Cloud.

        :param workspace: Optional workspace in Deepset Cloud
                          If None, the EvaluationSetClient's default workspace (self.workspace) will be used.

        :return: List of dictionaries that represent deepset Cloud evaluation sets.
                 These contain ("name", "evaluation_set_id", "created_at", "matched_labels", "total_labels") as fields.
        """
        evaluation_sets_response = self._get_evaluation_set(evaluation_set=None, workspace=workspace)

        return [eval_set for eval_set in evaluation_sets_response]

    def _get_evaluation_set(self, evaluation_set: Optional[str], workspace: Optional[str] = None) -> Generator:
        if not evaluation_set:
            evaluation_set = self.evaluation_set

        url = self._build_workspace_url(workspace=workspace)
        evaluation_set_url = f"{url}/evaluation_sets"

        for response in self.client.get_with_auto_paging(url=evaluation_set_url, query_params={"name": evaluation_set}):
            yield response

    def _get_labels_from_evaluation_set(
        self, workspace: Optional[str] = None, evaluation_set_id: Optional[str] = None
    ) -> Generator:
        url = f"{self._build_workspace_url(workspace=workspace)}/evaluation_sets/{evaluation_set_id}"
        labels = self.client.get(url=url).json()

        for label in labels:
            yield label

    def _build_workspace_url(self, workspace: Optional[str] = None):
        if workspace is None:
            workspace = self.workspace
        return self.client.build_workspace_url(workspace)


class DeepsetCloud:
    """
    A facade to communicate with Deepset Cloud.
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
        Creates a client to communicate with Deepset Cloud indexes.

        :param api_key: Secret value of the API key.
                        If not specified, will be read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the Deepset Cloud API.
                             If not specified, will be read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
        :param workspace: workspace in Deepset Cloud
        :param index: index in Deepset Cloud workspace

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
        Creates a client to communicate with Deepset Cloud pipelines.

        :param api_key: Secret value of the API key.
                        If not specified, will be read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the Deepset Cloud API.
                             If not specified, will be read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
        :param workspace: workspace in Deepset Cloud
        :param pipeline_config_name: name of the pipeline_config in Deepset Cloud workspace

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
        Creates a client to communicate with Deepset Cloud labels.

        :param api_key: Secret value of the API key.
                        If not specified, will be read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the Deepset Cloud API.
                             If not specified, will be read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
        :param workspace: workspace in Deepset Cloud
        :param evaluation_set: name of the evaluation set in Deepset Cloud

        """
        client = DeepsetCloudClient(api_key=api_key, api_endpoint=api_endpoint)
        return EvaluationSetClient(client=client, workspace=workspace, evaluation_set=evaluation_set)
