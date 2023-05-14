import os
import re
import sys
import base64
import argparse
import requests


VERSION_VALIDATOR = re.compile(r"^[0-9]+\.[0-9]+$")


class ReadmeAuth(requests.auth.AuthBase):
    def __call__(self, r):
        r.headers["authorization"] = f"Basic {readme_token()}"
        return r


def readme_token():
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


def create_new_unstable(current, new):
    """
    Create new version by copying current.

    :param current: Existing current unstable version
    :param new: Non existing new unstable version
    """
    url = "https://dash.readme.com/api/v1/version/"
    payload = {"is_beta": False, "version": new, "from": current, "is_hidden": False, "is_stable": False}
    res = requests.post(url, json=payload, auth=ReadmeAuth(), timeout=30)
    res.raise_for_status()


def promote_unstable_to_stable(unstable, stable):
    """
    Rename the current unstable to stable and set it as stable.

    :param unstable: Existing unstable version
    :param stable: Non existing new stable version
    """
    url = f"https://dash.readme.com/api/v1/version/{unstable}"
    payload = {"is_beta": False, "version": stable, "from": unstable, "is_hidden": False, "is_stable": True}
    res = requests.put(url, json=payload, auth=ReadmeAuth(), timeout=30)
    res.raise_for_status()


def calculate_new_unstable(version):
    # version must be formatted like so <major>.<minor>
    major, minor = version.split(".")
    return f"{major}.{int(minor) + 1}-unstable"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--new-version", help="The new minor version that is being released (e.g. 1.9).", required=True
    )
    args = parser.parse_args()

    if VERSION_VALIDATOR.match(args.new_version) is None:
        sys.exit("Version must be formatted like so <major>.<minor>")

    # This two are the version that we must have published in the end
    new_stable = f"{args.new_version}"
    new_unstable = calculate_new_unstable(args.new_version)

    versions = get_versions()
    new_stable_is_published = new_stable in versions
    new_unstable_is_published = new_unstable in versions

    if new_stable_is_published and new_unstable_is_published:
        # If both versions are published there's nothing to do.
        # We fail gracefully.
        print(f"Both new version {new_stable} and {new_unstable} are already published.")
        sys.exit(0)
    elif new_stable_is_published or new_unstable_is_published:
        # Either new stable or unstable is already published, it's to risky to
        # proceed so we abort the publishing process.
        sys.exit(f"Either version {new_stable} or {new_unstable} are already published. Too risky to proceed.")

    # This version must exist since it's the one we're trying to promote
    # to stable.
    current_unstable = f"{new_stable}-unstable"

    if current_unstable not in versions:
        sys.exit(f"Can't find version {current_unstable} to promote to {new_stable}")

    # First we create new unstable from the currently existing one
    create_new_unstable(current_unstable, new_unstable)
    # Then we promote the current unstable to stable since it's the one being published
    promote_unstable_to_stable(current_unstable, new_stable)
