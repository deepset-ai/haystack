"""
Increment the minor version of Haystack
 e.g. v1.10.0rc0 --> v1.11.0rc0
"""

with open("VERSION.txt") as f:
    version = f.read().strip()
version_split = version.split(".")
version_split[1] = str(int(version_split[1]) + 1)
new_version = ".".join(version_split)
with open("VERSION.txt", "w") as f:
    f.write(new_version)
