import requests
import base64
import os

from pprint import pprint


api_key = "rdme_xn8s9h3c87ac7deaaa300e4f33de07cd31b635b71dd9c25af1fca41ec7063c9383723f:"
api_key_b64 = "Basic " + base64.b64encode(api_key.encode("utf-8")).decode("utf-8")
print(api_key_b64)

curr_latest = "v6.0-latest"
new_version = "v6.0"


def assert_valid_version(new_version):
    if not new_version.startswith("v"):
        raise ValueError("Version must start with 'v'")
    if not new_version[1:].replace(".", "").replace("-latest", "").isdigit():
        raise ValueError("Version must be a number")
    return True


def get_versions():
    url = "https://dash.readme.com/api/v1/version"
    headers = {"Accept": "application/json", "Authorization": api_key_b64}
    response = requests.get(url, headers=headers)
    return [v["version"] for v in response.json()]


def create_version(new_version, fork_from_version, is_stable=False):
    url = "https://dash.readme.com/api/v1/version"
    payload = {
        "is_beta": False,
        "version": new_version,
        "from": fork_from_version,
        "is_hidden": False,
        "is_stable": is_stable,
    }
    headers = {"Accept": "application/json", "Content-Type": "application/json", "Authorization": api_key_b64}
    response = requests.post(url, json=payload, headers=headers)
    print("create_version()")
    print(response.text)


def delete_version(version_name):
    url = "https://dash.readme.com/api/v1/version/{version_name}".format(version_name=version_name)
    headers = {"Accept": "application/json", "Authorization": api_key_b64}
    response = requests.delete(url, headers=headers)
    print("delete_version()")
    print(response.text)


def update_version_name(old_latest_name, new_latest_name):
    url = "https://dash.readme.com/api/v1/version/{}".format(old_latest_name)
    payload = {
        "is_beta": False,
        "version": new_latest_name,
        "from": old_latest_name,
        "is_stable": True,
        "is_hidden": False,
    }

    headers = {"accept": "application/json", "content-type": "application/json", "authorization": api_key_b64}

    response = requests.put(url, json=payload, headers=headers)
    print(response.text)


def generate_new_latest_name(version):
    assert "-latest" not in version
    new_latest = version + "-latest"
    return new_latest


def get_category_id(version):
    import requests

    url = "https://dash.readme.com/api/v1/categories/haystack-rest-api"
    headers = {
        "accept": "application/json",
        "x-readme-version": version,
        "authorization": "Basic cmRtZV94bjhzOWgzYzg3YWM3ZGVhYWEzMDBlNGYzM2RlMDdjZDMxYjYzNWI3MWRkOWMyNWFmMWZjYTQxZWM3MDYzYzkzODM3MjNmOg==",
    }
    response = requests.get(url, headers=headers)
    return response.json()["id"]


def change_api_category_id(new_version, docs_dir):
    category_id = get_category_id(new_version)
    print(category_id)
    ## Replace the category id in the yaml headers
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".yml"):
                file_path = os.path.join(root, file)
                lines = [l for l in open(file_path, "r")]
                for l in lines:
                    if "category: " in l:
                        print("x")
                        lines[lines.index(l)] = "   category: {}\n".format(category_id)
                content = "".join(lines)
                with open(file_path, "w") as f:
                    f.write(content)


if __name__ == "__main__":
    versions = get_versions()
    # assert curr_latest[1:] in versions
    # assert new_version[1:] not in versions
    # create_version(new_version=new_version, fork_from_version=curr_latest)
    # new_latest_name = generate_new_latest_name(new_version)
    # update_version_name(curr_latest, new_latest_name)
    print(new_version)
    change_api_category_id("6.0", "_src/api/pydoc")


"""
1) Fork v1.0-latest to create v2.0

2) Rename v1.0 latest —> v2.0 latest
ensure v2.0-latest is the default version

3) Ensure latest tutorials and API to sync to v2.0-latest
- I assume we don’t need to do anything for this

4) Ensure v1.0 tutorials and API sync to v1.0 readme
- Get category IDs of v1.0 tutorials and API
- Modify headers in v1.0 tutorials and API
"""


"""
QUESTIONS
How do different keys work? API key vs authentication vs?
How to change category id in yamls?


"""
