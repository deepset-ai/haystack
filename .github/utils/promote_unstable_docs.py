import argparse
import re
import sys

from readme_api import get_versions, promote_unstable_to_stable

VERSION_VALIDATOR = re.compile(r"^[0-9]+\.[0-9]+$")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", help="The version to promote to stable (e.g. 2.1).", required=True)
    args = parser.parse_args()

    if VERSION_VALIDATOR.match(args.version) is None:
        sys.exit("Version must be formatted like so <major>.<minor>")

    unstable_version = f"{args.version}-unstable"
    stable_version = args.version

    versions = get_versions()
    if stable_version in versions:
        sys.exit(f"Version {stable_version} is already published.")

    if unstable_version not in versions:
        sys.exit(f"Can't find version {unstable_version} to promote to {stable_version}")

    promote_unstable_to_stable(unstable_version, stable_version)
