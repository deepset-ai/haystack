import re

from nbconvert import MarkdownExporter
import os
from pathlib import Path
from headers import headers


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    test = [ atoi(c) for c in re.split('(\d+)',text) ]
    return test


dir = Path("../../../../tutorials")

notebooks = [x for x in os.listdir(dir) if x[-6:] == ".ipynb"]
# sort notebooks based on numbers within name of notebook
notebooks = sorted(notebooks, key=lambda x: natural_keys(x))


e = MarkdownExporter(exclude_output=True)
for i, nb in enumerate(notebooks):
    body, resources = e.from_filename(dir / nb)
    with open(str(i + 1) + ".md", "w") as f:
        f.write(headers[i + 1] + "\n\n")
        f.write(body)

