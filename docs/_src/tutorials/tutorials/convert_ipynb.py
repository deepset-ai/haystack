from nbconvert import RSTExporter
import os
from pathlib import Path

dir = Path("../../../../tutorials")

notebooks = [x for x in os.listdir(dir) if x[-6:] == ".ipynb"]
notebooks = sorted(notebooks, key=lambda x: x[8])


e = RSTExporter(exclude_output=True)
for i, nb in enumerate(notebooks):
    body, resources = e.from_filename(dir / nb)
    with open(str(i + 1) + ".rst", "w") as f:
        f.write(body)

