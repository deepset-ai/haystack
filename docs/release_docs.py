import requests
import base64
import os

from pprint import pprint


api_key = "xxx"
api_key += ":"
api_key_b64 = "Basic " + base64.b64encode(api_key.encode("utf-8")).decode("utf-8")
print(api_key_b64)

new_version = "v5.1"

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


def update_version_name(old_unstable_name, new_unstable_name):
    url = "https://dash.readme.com/api/v1/version/{}".format(old_unstable_name)
    payload = {
        "is_beta": False,
        "version": new_unstable_name,
        "from": old_unstable_name,
        "is_hidden": False,
    }

    headers = {"accept": "application/json", "content-type": "application/json", "authorization": api_key_b64}

    response = requests.put(url, json=payload, headers=headers)
    print(response.text)


def generate_new_unstable_name(unstable_version_name):
    version_digits_str = unstable_version_name[1:].replace("-unstable", "")
    version_digits_split = version_digits_str.split(".")
    version_digits_split[1] = str(int(version_digits_split[1]) + 1)
    incremented_version_digits = ".".join(version_digits_split)
    new_unstable = "v" + incremented_version_digits + "-unstable"
    return new_unstable


def get_category_id(version):
    import requests

    url = "https://dash.readme.com/api/v1/categories/haystack-rest-api"
    headers = {
        "accept": "application/json",
        "x-readme-version": version,
        "authorization": api_key_b64,
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

def change_workflow(new_latest_name):
    # Change readme_integration.yml to use new latest version
    lines = [l for l in open("./../.github/workflows/readme_integration.yml", "r")]
    for l in lines:
        if "rdme: docs" in l:
            lines[lines.index(l)] = """          rdme: docs ./docs/_src/api/api/temp --key="$README_API_KEY" --version={}""".format(new_latest_name)
    content = "".join(lines)
    with open("./../.github/workflows/readme_integration.yml", "w") as f:
        f.write(content)

if __name__ == "__main__":
    versions = get_versions()
    curr_unstable = new_version + "-unstable"
    assert new_version[1:] not in versions
    assert curr_unstable[1:] in versions
    create_version(new_version=new_version, fork_from_version=curr_unstable, is_stable=False)
    new_unstable = generate_new_unstable_name(curr_unstable)
    update_version_name(curr_unstable, new_unstable)
    change_api_category_id(new_version, "_src/api/pydoc")
    change_workflow(new_unstable)
