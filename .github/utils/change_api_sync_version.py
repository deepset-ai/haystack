import argparse

README_INTEGRATION_WORKFLOW = "./.github/workflows/readme_api_sync.yml"

def change_workflow(new_latest_name, suffix_unstable=False):
    # Change readme_api_sync.yml to use new latest version
    lines = [l for l in open(README_INTEGRATION_WORKFLOW, "r")]
    if suffix_unstable:
        new_latest_name = new_latest_name + "-unstable"
    for l in lines:
        if "rdme: docs" in l:
            lines[lines.index(l)] = """          rdme: docs ./docs/_src/api/api/temp --key="$README_API_KEY" --version={}""".format(new_latest_name)
    content = "".join(lines)
    with open(README_INTEGRATION_WORKFLOW, "w") as f:
        f.write(content)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--version",
        help="The new minor version that is being released (e.g. v1.9).",
        required=True
    )
    parser.add_argument(
        "--unstable",
        help="Increment minor version by one and add --unstable suffix.",
        required=True
    )
    args = parser.parse_args()

    new_version = args.version
    # Drop the patch version, e.g. v1.9.1 -> v1.9
    split_version = new_version.split(".")
    if args.unstable:
        split_version[1] = str(int(split_version[1]) + 1)
    new_version = ".".join(new_version.split(".")[:2])

    # edit the version that the readme_api_sync.yml workflow uses (e.g. v1.9-unstable -> v1.10-unstable)
    change_workflow(new_version, suffix_unstable=args.unstable)