import requests
import pprint
import base64
import argparse
import os

PYDOC_CONFIGS_DIR = "./docs/_src/api/pydoc"

def get_category_id(version):
    url = "https://dash.readme.com/api/v1/categories/haystack-classes"
    headers = {
        "accept": "application/json",
        "x-readme-version": version,
        "authorization": api_key_b64,
    }
    ret = requests.get(url, headers=headers)
    pprint(ret.text)
    return ret.json()["id"]

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


if __name__ == "__main__":
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
    new_version = args.version

    api_key = args.key
    api_key += ":"
    api_key_b64 = "Basic " + base64.b64encode(api_key.encode("utf-8")).decode("utf-8")

    # edit the category id in the yaml headers of pydoc configs
    change_api_category_id(new_version, PYDOC_CONFIGS_DIR)

