import argparse
import sys
import os
import glob
import pathlib
import subprocess
from typing import Sequence

import yaml


def load_search_paths():
    """
    Load the search path for each pydoc configuration file and return
    a map {search_path -> config_file}
    """
    paths = dict()
    for fname in glob.glob("docs/_src/api/pydoc/*.yml"):
        with open(fname) as f:
            config = yaml.safe_load(f.read())
            # we always have only one loader in Haystack
            loader = config["loaders"][0]
            # `search_path` is a list but we always have only one item in Haystack
            search_path = loader["search_path"][0]
            # we only need the relative path from the root, let's call `resolve` to
            # get rid of the `../../` prefix
            search_path = str(pathlib.Path(search_path).resolve())
            # `resolve` will prepend a `/` to the path, remove it
            paths[search_path[1:]] = fname
    return paths


def main(argv: Sequence[str] = sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*", help="Filenames to check.")
    args = parser.parse_args(argv)

    search_paths = load_search_paths()

    for filename in args.filenames:
        # for each file in the commit queue, check if its path belongs
        # to any search path known to pydoc. If not, skip to the next one.
        search_path = str(pathlib.Path(filename).parent)
        if search_path not in search_paths:
            continue

        # if we get here, we need to build the docs: get the full path to the
        # config file and shell out `pydoc-markdown` after changing the current
        # working dir to the destination path.
        config_yml = os.path.abspath(pathlib.Path(search_paths[search_path]))
        res = subprocess.run(["pydoc-markdown", config_yml], cwd="docs/_src/api/api/")
        if res.returncode != 0:
            return res.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
