import base64
import argparse
import requests

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

def get_categories(version):
    url = "https://dash.readme.com/api/v1/categories?perPage=10&page=1"
    headers = {
        "accept": "application/json",
        "x-readme-version": version,
        "authorization": api_key_b64,
    }
    response = requests.get(url, headers=headers)
    return response.text


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
    args = parser.parse_args()

    api_key = args.key
    api_key += ":"
    api_key_b64 = "Basic " + base64.b64encode(api_key.encode("utf-8")).decode("utf-8")

    new_version = args.version
    # Drop the patch version, e.g. v1.9.1 -> v1.9
    new_version = ".".join(new_version.split(".")[:2])
    versions = get_versions()

    curr_unstable = new_version + "-unstable"
    assert new_version[1:] not in versions, "Version {} already exists in Readme.".format(new_version[1:])
    assert curr_unstable[1:] in versions, "Version {} does not exist in Readme.".format(curr_unstable[1:])

    # create v1.9 forked from v1.9-unstable
    create_version(new_version=new_version, fork_from_version=curr_unstable, is_stable=False)

    # rename v1.9-unstable to v1.10-unstable
    new_unstable = generate_new_unstable_name(curr_unstable)
    update_version_name(curr_unstable, new_unstable)


