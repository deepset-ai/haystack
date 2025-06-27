# /// script
# dependencies = [
#   "requests",
# ]
# ///

import os
import sys
import json
import argparse
from typing import Optional
from pathlib import Path

import requests


def transform_filename(filepath: Path) -> str:
    """
    Transform a file path to the required format:
    - Replace path separators with underscores
    """
    # Convert to string and replace path separators with underscores
    transformed = str(filepath).replace("/", "_").replace("\\", "_")

    return transformed


def upload_file_to_deepset(filepath: Path, api_key: str, workspace: str) -> bool:
    """
    Upload a single file to Deepset API.
    """
    # Read file content
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return False

    # Transform filename
    transformed_name = transform_filename(filepath)

    # Prepare metadata
    metadata: dict[str, str] = {"original_file_path": str(filepath)}

    # Prepare API request
    url = f"https://api.cloud.deepset.ai/api/v1/workspaces/{workspace}/files"
    params: dict[str, str] = {"file_name": transformed_name, "write_mode": "OVERWRITE"}

    headers: dict[str, str] = {
        "accept": "application/json",
        "authorization": f"Bearer {api_key}",
    }

    # Prepare multipart form data
    files: dict[str, tuple[None, str, str]] = {
        "meta": (None, json.dumps(metadata), "application/json"),
        "text": (None, content, "text/plain"),
    }

    try:
        response = requests.post(url, params=params, headers=headers, files=files, timeout=300)
        response.raise_for_status()
        print(f"Successfully uploaded: {filepath} as {transformed_name}")
        return True
    except requests.exceptions.HTTPError:
        print(f"Failed to upload {filepath}: HTTP {response.status_code}")
        print(f"   Response: {response.text}")
        return False
    except Exception as e:
        print(f"Failed to upload {filepath}: {e}")
        return False


def delete_files_from_deepset(
    filepaths: list[Path], api_key: str, workspace: str
) -> bool:
    """
    Delete multiple files from Deepset API.
    """
    if not filepaths:
        return True

    # Transform filenames
    transformed_names: list[str] = [transform_filename(fp) for fp in filepaths]

    # Prepare API request
    url = f"https://api.cloud.deepset.ai/api/v1/workspaces/{workspace}/files"

    headers: dict[str, str] = {
        "accept": "application/json",
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }

    data: dict[str, list[str]] = {"names": transformed_names}

    try:
        response = requests.delete(url, headers=headers, json=data, timeout=300)
        response.raise_for_status()
        print(f"Successfully deleted {len(transformed_names)} file(s):")
        for original, transformed in zip(filepaths, transformed_names):
            print(f"   - {original} (as {transformed})")
        return True
    except requests.exceptions.HTTPError:
        print(f"Failed to delete files: HTTP {response.status_code}")
        print(f"   Response: {response.text}")
        return False
    except Exception as e:
        print(f"Failed to delete files: {e}")
        return False


def main() -> None:
    """
    Main function to process and upload/delete files.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Upload/delete Python files to/from Deepset"
    )
    parser.add_argument(
        "--changed", nargs="*", default=[], help="Changed or added files"
    )
    parser.add_argument("--deleted", nargs="*", default=[], help="Deleted files")
    args = parser.parse_args()

    # Get environment variables
    api_key: Optional[str] = os.environ.get("DEEPSET_API_KEY")
    workspace: str = os.environ.get("DEEPSET_WORKSPACE")

    if not api_key:
        print("Error: DEEPSET_API_KEY environment variable not set")
        sys.exit(1)

    # Process arguments and convert to Path objects
    changed_files: list[Path] = [Path(f.strip()) for f in args.changed if f.strip()]
    deleted_files: list[Path] = [Path(f.strip()) for f in args.deleted if f.strip()]

    if not changed_files and not deleted_files:
        print("No files to process")
        sys.exit(0)

    print(f"Processing files in Deepset workspace: {workspace}")
    print("-" * 50)

    # Track results
    upload_success: int = 0
    upload_failed: list[Path] = []
    delete_success: bool = False

    # Handle deletions first
    if deleted_files:
        print(f"\nDeleting {len(deleted_files)} file(s)...")
        delete_success = delete_files_from_deepset(deleted_files, api_key, workspace)

    # Upload changed/new files
    if changed_files:
        print(f"\nUploading {len(changed_files)} file(s)...")
        for filepath in changed_files:
            if filepath.exists():
                if upload_file_to_deepset(filepath, api_key, workspace):
                    upload_success += 1
                else:
                    upload_failed.append(filepath)
            else:
                print(f"Skipping non-existent file: {filepath}")

    # Summary
    print("-" * 50)
    print("Processing Summary:")
    if changed_files:
        print(
            f"   Uploads - Successful: {upload_success}, Failed: {len(upload_failed)}"
        )
    if deleted_files:
        print(
            f"   Deletions - {'Successful' if delete_success else 'Failed'}: {len(deleted_files)} file(s)"
        )

    if upload_failed:
        print("\nFailed uploads:")
        for f in upload_failed:
            print(f"   - {f}")

    # Exit with error if any operation failed
    if upload_failed or (deleted_files and not delete_success):
        sys.exit(1)

    print("\nAll operations completed successfully!")


if __name__ == "__main__":
    main()
