#!/usr/bin/env python3
"""
Update the haystack-ai version in the deepset-cloud-custom-nodes uv.lock file.

Fetches sdist/wheel hashes from PyPI and updates the haystack-ai package entry,
preserving the existing lock file formatting.
"""

import argparse
import json
import sys
import urllib.request

import tomlkit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="Version to update to (e.g. 2.26.1-rc1)")
    parser.add_argument("lock_file", help="Path to uv.lock")
    args = parser.parse_args()

    # PEP 440 normalized version for filenames
    new_version = args.version.replace("-", "")

    # Fetch hashes from PyPI
    pypi_data = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/haystack-ai/{args.version}/json"))
    sdist_sha = wheel_sha = None
    for u in pypi_data["urls"]:
        if u["packagetype"] == "sdist":
            sdist_sha = u["digests"]["sha256"]
        elif u["packagetype"] == "bdist_wheel":
            wheel_sha = u["digests"]["sha256"]
    if not sdist_sha or not wheel_sha:
        sys.exit("Could not find sdist or wheel hashes on PyPI")

    with open(args.lock_file) as f:
        data = tomlkit.load(f)

    found = False
    for pkg in data["package"]:
        if pkg["name"] == "haystack-ai":
            old_version = pkg["version"]

            pkg["version"] = new_version

            pkg["sdist"]["url"] = pkg["sdist"]["url"].replace(old_version, new_version)
            pkg["sdist"]["hash"] = f"sha256:{sdist_sha}"

            wheel = pkg["wheels"][0]
            wheel["url"] = wheel["url"].replace(old_version, new_version)
            wheel["hash"] = f"sha256:{wheel_sha}"

            found = True
            print(f"Updated haystack-ai from {old_version} to {new_version}")
            break

    if not found:
        sys.exit("haystack-ai package not found in uv.lock")

    with open(args.lock_file, "w") as f:
        tomlkit.dump(data, f)
