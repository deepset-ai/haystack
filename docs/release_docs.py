import requests

api_key = "xxx"

latest = "v5.0.0-latest"
new_version = "v6.0.0"


def assert_valid_version(new_version):
    if not new_version.startswith("v"):
        raise ValueError("Version must start with 'v'")
    if not new_version[1:].replace(".", "").replace("-latest", "").isdigit():
        raise ValueError("Version must be a number")
    return True


def get_versions():
    url = "https://dash.readme.com/api/v1/version"
    headers = {
        "Accept": "application/json",
        "Authorization": "Basic cmRtZV94bjhzOWgzYzg3YWM3ZGVhYWEzMDBlNGYzM2RlMDdjZDMxYjYzNWI3MWRkOWMyNWFmMWZjYTQxZWM3MDYzYzkzODM3MjNmOg==",
    }
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
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Basic cmRtZV94bjhzOWgzYzg3YWM3ZGVhYWEzMDBlNGYzM2RlMDdjZDMxYjYzNWI3MWRkOWMyNWFmMWZjYTQxZWM3MDYzYzkzODM3MjNmOg==",
    }
    response = requests.post(url, json=payload, headers=headers)
    print("create_version()")
    print(response.text)


def delete_version(version_name):
    url = "https://dash.readme.com/api/v1/version/{version_name}".format(version_name=version_name)
    headers = {
        "Accept": "application/json",
        "Authorization": "Basic cmRtZV94bjhzOWgzYzg3YWM3ZGVhYWEzMDBlNGYzM2RlMDdjZDMxYjYzNWI3MWRkOWMyNWFmMWZjYTQxZWM3MDYzYzkzODM3MjNmOg==",
    }
    response = requests.delete(url, headers=headers)
    print("delete_version()")
    print(response.text)


def update_version_name(old_latest_name, new_latest_name):
    # create new latest
    create_version(new_version=new_latest_name, fork_from_version=old_latest_name)
    # delete old latest
    delete_version(old_latest_name)


def generate_new_latest_name(latest):
    assert "-latest" in latest


from pprint import pprint

versions = get_versions()
pprint(versions)
assert latest[1:] in versions
assert new_version[1:] not in versions
create_version(new_version=new_version, fork_from_version=latest)
new_latest_name = generate_new_latest_name(latest)
update_version_name(old_latest_name, new_latest_name)


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
