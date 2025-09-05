import argparse
import re
import sys

from readme_api import create_new_unstable, get_versions

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

    # This two are the version that we must have published in the end
    new_stable = f"{args.new_version}"
    new_unstable = calculate_new_unstable(args.new_version)

    versions = get_versions()
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

    # This version must exist since it's the one we're trying to promote
    # to stable.
    current_unstable = f"{new_stable}-unstable"

    if current_unstable not in versions:
        sys.exit(f"Can't find version {current_unstable} to promote to {new_stable}")

    # Create create new unstable from the currently existing one.
    # The new unstable will be made stable at a later time by another workflow
    create_new_unstable(current_unstable, new_unstable)
