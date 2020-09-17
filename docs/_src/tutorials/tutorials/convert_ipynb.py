from nbconvert import MarkdownExporter
import os
from pathlib import Path
from headers import headers

dir = Path("../../../../tutorials")

notebooks = [x for x in os.listdir(dir) if x[-6:] == ".ipynb"]
notebooks = sorted(notebooks, key=lambda x: x[8])


e = MarkdownExporter(exclude_output=True)
for i, nb in enumerate(notebooks):
    body, resources = e.from_filename(dir / nb)
    with open(str(i + 1) + ".md", "w") as f:
        f.write(headers[i + 1] + "\n\n")
        f.write(body)

