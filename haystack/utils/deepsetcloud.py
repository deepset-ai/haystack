import logging
import os
from typing import Optional
import requests

DEFAULT_API_ENDPOINT = f"DC_API_PLACEHOLDER/v1" #TODO

logger = logging.getLogger(__name__)


class BearerAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token
    def __call__(self, r):
        r.headers["authorization"] = "Bearer " + self.token
        return r

class DeepsetCloudAdapter:
    def __init__(        
        self, 
        api_key: str = None, 
        api_endpoint: Optional[str] = None,
        workspace: Optional[str] = None,
        index: Optional[str] = None,
        pipeline: Optional[str] = None):
        """
        An adapter to communicate with Deepset Cloud.

        :param api_key: Secret value of the API key. 
                        If not specified, will be read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the Deepset Cloud API. 
                             If not specified, will be read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
        :param workspace: workspace in Deepset Cloud

        """
        self.api_key = api_key or os.getenv("DEEPSET_CLOUD_API_KEY")
        if self.api_key is None:
            raise ValueError("No api_key specified. Please set api_key param or DEEPSET_CLOUD_API_KEY environment variable.")

        self.api_endpoint = api_endpoint or os.getenv("DEEPSET_CLOUD_API_ENDPOINT", DEFAULT_API_ENDPOINT)
        self._workspace = workspace
        self._index = index
        self._pipeline = pipeline

    def workspace(self, workspace: Optional[str] = None):
        return self.clone(workspace=workspace)

    def index(self, index: Optional[str] = None):
        return self.clone(index=index)

    def pipeline(self, pipeline: Optional[str] = None):
        return self.clone(pipeline=pipeline)
    
    def clone(self, api_key: str = None, api_endpoint: str = None, 
                                    workspace: str = None, index: str = None, pipeline: str = None):
        api_key = api_key or self.api_key
        api_endpoint = api_endpoint or self.api_endpoint
        workspace = workspace or self._workspace
        index = index or self._index
        pipeline = pipeline or self._pipeline
        return DeepsetCloudAdapter(api_key=api_key, api_endpoint=api_endpoint, workspace=workspace, index=index, pipeline=pipeline)
    
    def get(self, relative_path: str = None, headers: dict = None, query_params: dict = None):
        url = self._build_url(relative_path=relative_path)
        return requests.get(url=url, auth=BearerAuth(self.api_key), headers=headers, params=query_params)

    def post(self, relative_path: str = None, json: dict = {}, stream: bool = False, headers: dict = None):
        url = self._build_url(relative_path=relative_path)
        return requests.post(url=url, json=json, stream=stream, headers=headers, auth=BearerAuth(self.api_key))

    def _build_url(self, relative_path: str = None):        
        api_endpoint = self.api_endpoint.rstrip("/")
        url = api_endpoint
        if self._workspace is not None:
            url = f"{url}/workspaces/{self._workspace}"
        if self._index and self._pipeline:
            raise ValueError("Both index and pipeline are set. Please set only one of them.")
        elif self._index is not None:
            url = f"{url}/indexes/{self._index}"
        elif self._pipeline is not None:
            url = f"{url}/pipelines/{self._pipeline}"
        if relative_path is not None:
            relative_path = relative_path.lstrip("/")
            url = f"{url}/{relative_path}"
        return url
