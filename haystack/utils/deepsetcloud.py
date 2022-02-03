import logging
import os
from typing import Dict, List, Optional
import requests

DEFAULT_API_ENDPOINT = f"DC_API_PLACEHOLDER/v1"  # TODO

logger = logging.getLogger(__name__)


class BearerAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token

    def __call__(self, r):
        r.headers["authorization"] = "Bearer " + self.token
        return r


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
            raise ValueError(
                "No api_key specified. Please set api_key param or DEEPSET_CLOUD_API_KEY environment variable."
            )

        self.api_endpoint = api_endpoint or os.getenv("DEEPSET_CLOUD_API_ENDPOINT", DEFAULT_API_ENDPOINT)

    def get(self, url: str, headers: dict = None, query_params: dict = None, raise_on_error: bool = True):
        response = requests.get(url=url, auth=BearerAuth(self.api_key), headers=headers, params=query_params)
        if raise_on_error and response.status_code > 299:
            raise Exception(
                f"GET {url} failed: HTTP {response.status_code} - {response.reason}\n{response.content.decode()}"
            )
        return response

    def post(self, url: str, json: dict = {}, stream: bool = False, headers: dict = None, raise_on_error: bool = True):
        json = self._remove_null_values(json)
        response = requests.post(url=url, json=json, stream=stream, headers=headers, auth=BearerAuth(self.api_key))
        if raise_on_error and response.status_code > 299:
            raise Exception(
                f"POST {url} failed: HTTP {response.status_code} - {response.reason}\n{response.content.decode()}"
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
            raise Exception(f"Could not connect to Deepset Cloud:\n{ie}") from ie

    def query(
        self,
        query: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        top_k: int = 10,
        custom_query: Optional[str] = None,
        query_emb: Optional[List[float]] = None,
        return_embedding: Optional[bool] = None,
        similarity: Optional[str] = None,
        workspace: Optional[str] = None,
        index: Optional[str] = None,
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
        response = self.client.get(url=pipeline_config_url, headers=headers)
        return response.json()

    def _build_pipeline_url(self, workspace: Optional[str] = None, pipeline_config_name: Optional[str] = None):
        if workspace is None:
            workspace = self.workspace
        if pipeline_config_name is None:
            pipeline_config_name = self.pipeline_config_name
        workspace_url = self.client.build_workspace_url(workspace)
        return f"{workspace_url}/pipelines/{pipeline_config_name}"


class DeepsetCloud:
    """
    A facade to communicate with Deepset Cloud.
    """

    @classmethod
    def get_index_client(
        cls,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        workspace: Optional[str] = None,
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
        workspace: Optional[str] = None,
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
