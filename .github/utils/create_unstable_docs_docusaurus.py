"""
This script creates an unstable documentation version at the time of branch-off for a new Haystack release.

Between branch-off and the actual release, two unstable doc versions coexist.
If we branch off for 2.20, we have:
1. the target unstable version, 2.20-unstable (lives in docs-website/versioned_docs/version-2.20-unstable)
2. the next unstable version, 2.21-unstable (lives in docs-website/docs)

This script takes care of all the necessary updates to the documentation website.
"""

import argparse
import json
import os
import re
import shutil
import sys

VERSION_VALIDATOR = re.compile(r"^[0-9]+\.[0-9]+$")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--new-version", help="The new unstable version that is being created (e.g. 2.20).", required=True
    )
    args = parser.parse_args()

    if VERSION_VALIDATOR.match(args.new_version) is None:
        sys.exit("Version must be formatted like so <major>.<minor>")

    target_version = f"{args.new_version}"  # e.g., "2.20" - the target release version
    major, minor = args.new_version.split(".")

    target_unstable = f"{target_version}-unstable"  # e.g., "2.20-unstable"
    next_unstable = f"{major}.{int(minor) + 1}-unstable"  # e.g., "2.21-unstable" - next cycle
    previous_stable = f"{major}.{int(minor) - 1}"  # e.g., "2.19" - previous stable release

    versions = [
        folder.replace("version-", "")
        for folder in os.listdir("docs-website/versioned_docs")
        if os.path.isdir(os.path.join("docs-website/versioned_docs", folder))
    ]

    # Check if the versions we're about to create already exist in versioned_docs
    if target_version in versions:
        sys.exit(f"{target_version} already exists (already released). Aborting.")
    if target_unstable in versions:
        print(f"{target_unstable} already exists. Nothing to do.")
        sys.exit(0)

    # Create new unstable from the currently existing one.
    # The new unstable will be made stable at a later time by another workflow
    print(f"Creating new unstable version {target_unstable} from main")

    ### Docusaurus updates

    # copy docs to versioned_docs/version-target_unstable
    shutil.copytree("docs-website/docs", f"docs-website/versioned_docs/version-{target_unstable}")

    # copy reference to reference_versioned_docs/version-target_unstable
    shutil.copytree("docs-website/reference", f"docs-website/reference_versioned_docs/version-{target_unstable}")

    # copy versioned_sidebars/version-previous_stable-sidebars.json
    # to versioned_sidebars/version-target_unstable-sidebars.json
    shutil.copy(
        f"docs-website/versioned_sidebars/version-{previous_stable}-sidebars.json",
        f"docs-website/versioned_sidebars/version-{target_unstable}-sidebars.json",
    )

    # copy reference_versioned_sidebars/version-previous_stable-sidebars.json
    # to reference_versioned_sidebars/version-target_unstable-sidebars.json
    shutil.copy(
        f"docs-website/reference_versioned_sidebars/version-{previous_stable}-sidebars.json",
        f"docs-website/reference_versioned_sidebars/version-{target_unstable}-sidebars.json",
    )

    # add unstable version to versions.json
    with open("docs-website/versions.json", "r") as f:
        versions_list = json.load(f)
    versions_list.insert(0, target_unstable)
    with open("docs-website/versions.json", "w") as f:
        json.dump(versions_list, f)

    # add unstable version to reference_versions.json
    with open("docs-website/reference_versions.json", "r") as f:
        reference_versions_list = json.load(f)
    reference_versions_list.insert(0, target_unstable)
    with open("docs-website/reference_versions.json", "w") as f:
        json.dump(reference_versions_list, f)

    # in docusaurus.config.js, replace the target unstable version with the next unstable version
    with open("docs-website/docusaurus.config.js", "r") as f:
        config = f.read()
    config = config.replace(f"label: '{target_unstable}'", f"label: '{next_unstable}'")
    with open("docs-website/docusaurus.config.js", "w") as f:
        f.write(config)
