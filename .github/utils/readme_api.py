import base64
import os

import requests


class ReadmeAuth(requests.auth.AuthBase):
    """
    Custom authentication class for Readme API.
    """

    def __call__(self, r):  # noqa: D102
        r.headers["authorization"] = f"Basic {readme_token()}"
        return r


def readme_token():
    """
    Get the Readme API token from the environment variable and encode it in base64.
    """
    api_key = os.getenv("RDME_API_KEY", None)
    if not api_key:
        raise Exception("RDME_API_KEY env var is not set")

    api_key = f"{api_key}:"
    return base64.b64encode(api_key.encode("utf-8")).decode("utf-8")


def get_versions():
    """
    Return all versions currently published in Readme.io.
    """
    url = "https://dash.readme.com/api/v1/version"
    res = requests.get(url, auth=ReadmeAuth(), timeout=30)
    res.raise_for_status()
    return [v["version"] for v in res.json()]


def create_new_unstable(current: str, new: str):
    """
    Create new version by copying current.

    :param current: Existing current unstable version
    :param new: Non existing new unstable version
    """
    url = "https://dash.readme.com/api/v1/version/"
    payload = {"is_beta": False, "version": new, "from": current, "is_hidden": False, "is_stable": False}
    res = requests.post(url, json=payload, auth=ReadmeAuth(), timeout=30)
    res.raise_for_status()


def promote_unstable_to_stable(unstable: str, stable: str):
    """
    Rename the current unstable to stable and set it as stable.

    :param unstable: Existing unstable version
    :param stable: Non existing new stable version
    """
    url = f"https://dash.readme.com/api/v1/version/{unstable}"
    payload = {"is_beta": False, "version": stable, "from": unstable, "is_hidden": False, "is_stable": True}
    res = requests.put(url, json=payload, auth=ReadmeAuth(), timeout=30)
    res.raise_for_status()
