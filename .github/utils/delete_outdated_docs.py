import argparse
import base64
import os
import re
from pathlib import Path
from typing import List

import requests
import yaml

VERSION_VALIDATOR = re.compile(r"^[0-9]+\.[0-9]+$")


def readme_token():
    api_key = os.getenv("README_API_KEY", None)
    if not api_key:
        raise Exception("README_API_KEY env var is not set")

    api_key = f"{api_key}:"
    return base64.b64encode(api_key.encode("utf-8")).decode("utf-8")


def create_headers(version: str):
    return {"authorization": f"Basic {readme_token()}", "x-readme-version": version}


def get_docs_in_category(category_slug: str, version: str) -> List[str]:
    """
    Returns the slugs of all documents in a category for the specific version.
    """
    url = f"https://dash.readme.com/api/v1/categories/{category_slug}/docs"
    headers = create_headers(version)
    res = requests.get(url, headers=headers, timeout=10)
    return [doc["slug"] for doc in res.json()]


def delete_doc(slug: str, version: str):
    url = f"https://dash.readme.com/api/v1/docs/{slug}"
    headers = create_headers(version)
    res = requests.delete(url, headers=headers, timeout=10)
    res.raise_for_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete outdated documentation from Readme.io. "
        "It will delete all documents that are not present in the current config files."
    )
    parser.add_argument(
        "-c", "--config-path", help="Path to folder containing YAML documentation configs", required=True, type=Path
    )
    parser.add_argument("-v", "--version", help="The version that will have its documents deleted", required=True)
    args = parser.parse_args()

    configs = [yaml.safe_load(c.read_text()) for c in args.config_path.glob("*.yml")]

    remote_docs = {}
    for config in configs:
        category_slug = config["renderer"]["category_slug"]
        if category_slug in remote_docs:
            continue
        docs = get_docs_in_category(category_slug, args.version)

        remote_docs[category_slug] = docs

    for config in configs:
        doc_slug = config["renderer"]["slug"]
        category_slug = config["renderer"]["category_slug"]
        if doc_slug in remote_docs[category_slug]:
            continue

        delete_doc(doc_slug, args.version)
