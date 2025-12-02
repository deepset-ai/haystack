"""
This script syncs the Haystack docs HTML files to the deepset workspace for search indexing.

It is used in the docs_search_sync.yml workflow.

1. Collects all HTML files from the docs and reference directories for the stable Haystack version.
2. Uploads the HTML files to the deepset workspace.
    - A timestamp-based metadata field is used to track document versions in the workspace.
3. Deletes the old HTML files from the deepset workspace.
    - Since most files are overwritten during upload, only a small number of deletions is expected.
    - In case MAX_DELETIONS_SAFETY_LIMIT is exceeded, we block the deletion.
"""

import os
import sys
import time
from pathlib import Path

import requests
from deepset_cloud_sdk.workflows.sync_client.files import DeepsetCloudFile, WriteMode, list_files, upload_texts

DEEPSET_WORKSPACE_DOCS_SEARCH = os.environ["DEEPSET_WORKSPACE_DOCS_SEARCH"]
DEEPSET_API_KEY_DOCS_SEARCH = os.environ["DEEPSET_API_KEY_DOCS_SEARCH"]

# If there are more files to delete than this limit, it's likely that something went wrong in the upload process.
MAX_DELETIONS_SAFETY_LIMIT = 20


def collect_docs_files(version: int) -> list[DeepsetCloudFile]:
    """
    Collect all HTML files from the docs and reference directories.

    Returns a list of DeepsetCloudFile objects.
    """
    repo_root = Path(__file__).parent.parent.parent
    build_dir = repo_root / "docs-website" / "build"
    # we want to exclude previous and temporarily unstable versions (2.x) and next version (next)
    exclude = ("2.", "next")

    files = []
    for section in ("docs", "reference"):
        for subfolder in (build_dir / section).iterdir():
            if subfolder.is_dir() and not any(x in subfolder.name for x in exclude):
                for html_file in subfolder.rglob("*.html"):
                    files.append(
                        DeepsetCloudFile(
                            # The build produces files like docs/agents/index.html or reference/agents-api/index.html.
                            # For file names, we want to use the parent directory name (agents.html or agents-api.html)
                            name=f"{html_file.parent.name}.html",
                            text=html_file.read_text(),
                            meta={
                                "type": "api-reference" if section == "reference" else "documentation",
                                "version": version,
                            },
                        )
                    )
    return files


def delete_files(file_names: list[str]) -> None:
    """
    Delete files from the deepset workspace.
    """
    url = f"https://api.cloud.deepset.ai/api/v1/workspaces/{DEEPSET_WORKSPACE_DOCS_SEARCH}/files"
    payload = {"names": file_names}
    headers = {"Accept": "application/json", "Authorization": f"Bearer {DEEPSET_API_KEY_DOCS_SEARCH}"}
    response = requests.delete(url, json=payload, headers=headers, timeout=300)
    response.raise_for_status()


if __name__ == "__main__":
    version = time.time_ns()
    print(f"Docs version: {version}")

    print("Collecting docs files from build directory")
    dc_files = collect_docs_files(version)
    print(f"Collected {len(dc_files)} docs files")

    if len(dc_files) == 0:
        print("No docs files found. Something is wrong. Exiting.")
        sys.exit(1)

    print("Uploading docs files to deepset")
    summary = upload_texts(
        workspace_name=DEEPSET_WORKSPACE_DOCS_SEARCH,
        files=dc_files,
        api_key=DEEPSET_API_KEY_DOCS_SEARCH,
        blocking=True,  # Very important to ensure that DC is up to date when we query for deletion
        timeout_s=300,
        show_progress=True,
        write_mode=WriteMode.OVERWRITE,
        enable_parallel_processing=True,
    )
    print(f"Uploaded docs files to deepset\n{summary}")
    if summary.failed_upload_count > 0:
        print("Failed to upload some docs files. Stopping to prevent risky deletion of old files.")
        sys.exit(1)

    print("Listing old docs files from deepset")
    odata_filter = f"version lt '{version}'"
    old_files_names = [
        f.name
        for batch in list_files(
            workspace_name=DEEPSET_WORKSPACE_DOCS_SEARCH, api_key=DEEPSET_API_KEY_DOCS_SEARCH, odata_filter=odata_filter
        )
        for f in batch
    ]

    print(f"Found {len(old_files_names)} old files to delete")
    if len(old_files_names) > MAX_DELETIONS_SAFETY_LIMIT:
        print(
            f"Found >{MAX_DELETIONS_SAFETY_LIMIT} old files to delete. "
            "Stopping because something could have gone wrong in the upload process."
        )
        sys.exit(1)

    if len(old_files_names) > 0:
        print("Deleting old docs files from deepset")
        delete_files(old_files_names)
        print("Deleted old docs files from deepset")
