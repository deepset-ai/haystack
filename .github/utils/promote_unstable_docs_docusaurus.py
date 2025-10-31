"""
This script promotes an unstable documentation version to a stable version at the time of a new Haystack release.

To understand how unstable doc versions are created, see create_unstable_docs_docusaurus.py.
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
    parser.add_argument("-v", "--version", help="The version to promote to stable (e.g. 2.20).", required=True)
    args = parser.parse_args()

    if VERSION_VALIDATOR.match(args.version) is None:
        sys.exit("Version must be formatted like so <major>.<minor>")

    target_version = f"{args.version}"  # e.g., "2.20" - the target release version
    major, minor = args.version.split(".")

    target_unstable = f"{target_version}-unstable"  # e.g., "2.20-unstable"
    previous_stable = f"{major}.{int(minor) - 1}"  # e.g., "2.19" - previous stable release

    versions = [
        folder.replace("version-", "")
        for folder in os.listdir("docs-website/versioned_docs")
        if os.path.isdir(os.path.join("docs-website/versioned_docs", folder))
    ]

    if target_version in versions:
        sys.exit(f"{target_version} already exists (already released). Aborting.")
    if not target_unstable in versions:
        sys.exit(f"Can't find version {target_unstable} to promote to {target_version}")

    print(f"Promoting unstable version {target_unstable} to stable version {target_version}")

    ### Docusaurus updates

    # rename versioned_docs/version-target_unstable to versioned_docs/version-target_version
    shutil.move(
        f"docs-website/versioned_docs/version-{target_unstable}",
        f"docs-website/versioned_docs/version-{target_version}",
    )

    # rename reference_versioned_docs/version-target_unstable to reference_versioned_docs/version-target_version
    shutil.move(
        f"docs-website/reference_versioned_docs/version-{target_unstable}",
        f"docs-website/reference_versioned_docs/version-{target_version}",
    )

    # copy versioned_sidebars/version-target_unstable-sidebars.json
    # to versioned_sidebars/version-target_version-sidebars.json
    shutil.move(
        f"docs-website/versioned_sidebars/version-{target_unstable}-sidebars.json",
        f"docs-website/versioned_sidebars/version-{target_version}-sidebars.json",
    )

    # rename reference_versioned_sidebars/version-target_unstable-sidebars.json
    # to reference_versioned_sidebars/version-target_version-sidebars.json
    shutil.move(
        f"docs-website/reference_versioned_sidebars/version-{target_unstable}-sidebars.json",
        f"docs-website/reference_versioned_sidebars/version-{target_version}-sidebars.json",
    )

    # replace unstable version with stable version in versions.json
    with open("docs-website/versions.json", "r") as f:
        versions_list = json.load(f)
    versions_list[versions_list.index(target_unstable)] = target_version
    with open("docs-website/versions.json", "w") as f:
        json.dump(versions_list, f)

    # replace unstable version with stable version in reference_versions.json
    with open("docs-website/reference_versions.json", "r") as f:
        reference_versions_list = json.load(f)
    reference_versions_list[reference_versions_list.index(target_unstable)] = target_version
    with open("docs-website/reference_versions.json", "w") as f:
        json.dump(reference_versions_list, f)

    # in docusaurus.config.js, replace previous stable version with the target version
    with open("docs-website/docusaurus.config.js", "r") as f:
        config = f.read()
    config = config.replace(f"lastVersion: '{previous_stable}'", f"lastVersion: '{target_version}'")  # "2.19" -> "2.20"
    with open("docs-website/docusaurus.config.js", "w") as f:
        f.write(config)
