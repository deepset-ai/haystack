import os

from deepset_cloud_sdk.workflows.sync_client.files import list_files

files = list_files(workspace_name=os.getenv("WORKSPACE"), api_key=os.getenv("API_KEY"))
for file_batch in files:
    print(file_batch)
