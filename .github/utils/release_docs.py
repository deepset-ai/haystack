import requests
import base64
import os
import argparse
import requests

from pprint import pprint


README_INTEGRATION_WORKFLOW = "./.github/workflows/readme_api_sync.yml"
PYDOC_CONFIGS_DIR = "./docs/_src/api/pydoc"


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
    url = "https://dash.readme.com/api/v1/categories/haystack-classes"
    headers = {
        "accept": "application/json",
        "x-readme-version": version,
        "authorization": api_key_b64,
    }
    response = requests.get(url, headers=headers)
    pprint(response.text)
    return response.json()["id"]

def get_categories(version):
    url = "https://dash.readme.com/api/v1/categories?perPage=10&page=1"
    headers = {
        "accept": "application/json",
        "x-readme-version": version,
        "authorization": api_key_b64,
    }
    response = requests.get(url, headers=headers)
    return response.text


def change_api_category_id(new_version, docs_dir):
    print(new_version)
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
                        lines[lines.index(l)] = "   category: {}\n".format(category_id)
                content = "".join(lines)
                with open(file_path, "w") as f:
                    f.write(content)

def change_workflow(new_latest_name):
    # Change readme_api_sync.yml to use new latest version
    lines = [l for l in open(README_INTEGRATION_WORKFLOW, "r")]
    for l in lines:
        if "rdme: docs" in l:
            lines[lines.index(l)] = """          rdme: docs ./docs/_src/api/api/temp --key="$README_API_KEY" --version={}""".format(new_latest_name)
    content = "".join(lines)
    with open(README_INTEGRATION_WORKFLOW, "w") as f:
        f.write(content)

def hide_version(depr_version):
    url = "https://dash.readme.com/api/v1/version/{}".format(depr_version)
    payload = {
        "is_beta": False,
        "version": depr_version,
        "from": "",
        "is_hidden": True,
    }

    headers = {"accept": "application/json", "content-type": "application/json", "authorization": api_key_b64}

    response = requests.put(url, json=payload, headers=headers)
    print(response.text)

def generate_new_depr_name(depr_name):
    version_digits_str = depr_name[1:]
    version_digits_split = version_digits_str.split(".")
    version_digits_split[1] = str(int(version_digits_split[1]) + 1)
    incremented_version_digits = ".".join(version_digits_split)
    new_depr = "v" + incremented_version_digits + "-and-older"
    return new_depr

def get_old_and_older_name(versions):
    ret = []
    for v in versions:
        if v.endswith("-and-older"):
            ret.append(v)
    if len(ret) == 1:
        return ret[0]
    return None

def generate_new_and_older_name(old):
    digits_str = old[1:].replace("-and-older", "")
    digits_split = digits_str.split(".")
    digits_split[1] = str(int(digits_split[1]) + 1)
    incremented_digits = ".".join(digits_split)
    new = "v" + incremented_digits + "-and-older"
    return new

if __name__ == "__main__":
    # Comments below are for a case where we are releasing new_version="v1.9".
    # This requires for v1.9-unstable and v1.8 to exist in Readme.

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--version",
        help="The new minor version that is being released (e.g. v1.9.1).",
        required=True
    )
    parser.add_argument(
        "-k",
        "--key",
        help="The Readme API key for Haystack documentation.",
        required=True
    )
    parser.add_argument(
        "--skip_readme_changes",
        help="Do not perform any of the Readme release steps, only change the headers of the API docs so they point to a new category. Used for debugging.",
        action='store_true'
    )
    args = parser.parse_args()

    api_key = args.key
    api_key += ":"
    api_key_b64 = "Basic " + base64.b64encode(api_key.encode("utf-8")).decode("utf-8")

    new_version = args.version
    # Drop the patch version, e.g. v1.9.1 -> v1.9
    new_version = ".".join(new_version.split(".")[:2])
    versions = get_versions()

    if not args.skip_readme_changes:

        curr_unstable = new_version + "-unstable"
        assert new_version[1:] not in versions, "Version {} already exists in Readme.".format(new_version[1:])
        assert curr_unstable[1:] in versions, "Version {} does not exist in Readme.".format(curr_unstable[1:])

        # create v1.9 forked from v1.9-unstable
        create_version(new_version=new_version, fork_from_version=curr_unstable, is_stable=False)

        # rename v1.9-unstable to v1.10-unstable
        new_unstable = generate_new_unstable_name(curr_unstable)
        update_version_name(curr_unstable, new_unstable)

    # edit the category id in the yaml headers of pydoc configs
    change_api_category_id(new_version, PYDOC_CONFIGS_DIR)

    # edit the version that the readme_api_sync.yml workflow uses (e.g. v1.9-unstable -> v1.9)
    change_workflow(new_version)

    # ## hide v1.4 and rename v1.3-and-older to v1.4-and-older
    # old_and_older_name = "v" + get_old_and_older_name(versions)
    # new_and_older_name = generate_new_and_older_name(old_and_older_name)
    # depr_version = new_and_older_name.replace("-and-older", "")
    # hide_version(depr_version)
    # update_version_name(old_and_older_name, new_and_older_name)

