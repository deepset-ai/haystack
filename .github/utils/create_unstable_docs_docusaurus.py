import argparse
import json
import os
import re
import shutil
import sys

VERSION_VALIDATOR = re.compile(r"^[0-9]+\.[0-9]+$")


def calculate_new_unstable(version: str):
    """
    Calculate the new unstable version based on the given version.
    """
    # version must be formatted like so <major>.<minor>
    major, minor = version.split(".")
    return f"{major}.{int(minor) + 1}-unstable"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--new-version", help="The new unstable version that is being created (e.g. 1.9).", required=True
    )
    args = parser.parse_args()

    if VERSION_VALIDATOR.match(args.new_version) is None:
        sys.exit("Version must be formatted like so <major>.<minor>")

    # These two are the version that we must have published in the end
    new_stable = f"{args.new_version}"
    new_unstable = calculate_new_unstable(args.new_version)

    versions = [
        folder.replace("version-", "")
        for folder in os.listdir("docs-website/versioned_docs")
        if os.path.isdir(os.path.join("docs-website/versioned_docs", folder))
    ]

    print(f"Versions: {versions}")

    new_stable_is_published = new_stable in versions
    new_unstable_is_published = new_unstable in versions

    if new_stable_is_published and new_unstable_is_published:
        # If both versions are published there's nothing to do.
        # We fail gracefully.
        print(f"Both new version {new_stable} and {new_unstable} are already published.")
        sys.exit(0)
    elif new_stable_is_published or new_unstable_is_published:
        # Either new stable or unstable is already published, it's to risky to
        # proceed so we abort the publishing process.
        sys.exit(f"Either version {new_stable} or {new_unstable} are already published. Too risky to proceed.")

    # Create create new unstable from the currently existing one.
    # The new unstable will be made stable at a later time by another workflow
    print(f"Creating new unstable version {new_unstable} from main")

    # MANUAL DOCUSAURUS UPDATES

    # copy docs to versioned_docs/version-new_unstable
    shutil.copytree("docs-website/docs", f"docs-website/versioned_docs/version-{new_unstable}")

    # copy reference to versioned_docs/version-new_unstable/reference
    shutil.copytree("docs-website/reference", f"docs-website/versioned_docs/version-{new_unstable}/reference")

    # copy sidebars.js to versioned_docs/version-new_unstable/sidebars.js
    shutil.copy("docs-website/sidebars.js", f"docs-website/versioned_docs/version-{new_unstable}/sidebars.js")

    # copy reference-sidebars.js to versioned_docs/version-new_unstable/reference-sidebars.js
    shutil.copy(
        "docs-website/reference-sidebars.js",
        f"docs-website/versioned_docs/version-{new_unstable}/reference-sidebars.js",
    )

    # add unstable version to versions.json
    with open("docs-website/versions.json", "r") as f:
        versions = json.load(f)
    versions.insert(0, new_unstable)
    with open("docs-website/versions.json", "w") as f:
        json.dump(versions, f)

    # add unstable version to reference_versions.json
    with open("docs-website/reference_versions.json", "r") as f:
        reference_versions = json.load(f)
    reference_versions.insert(0, new_unstable)
    with open("docs-website/reference_versions.json", "w") as f:
        json.dump(reference_versions, f)

    # in docusaurus.config.js, replace the current version with the new unstable version
    current_version = f"{new_stable}-unstable"
    with open("docs-website/docusaurus.config.js", "r") as f:
        config = f.read()
    config = config.replace(current_version, new_unstable)
    with open("docs-website/docusaurus.config.js", "w") as f:
        f.write(config)
