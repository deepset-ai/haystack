#!/usr/bin/env python3

import re
from pathlib import Path
import argparse
import sys
import pathlib
from typing import Sequence

from nbconvert import MarkdownExporter


headers = {
    1: """<!---
title: "Tutorial 1"
metaTitle: "Build Your First QA System"
metaDescription: ""
slug: "/docs/tutorial1"
date: "2020-09-03"
id: "tutorial1md"
--->""",
    2: """<!---
title: "Tutorial 2"
metaTitle: "Fine-tuning a model on your own data"
metaDescription: ""
slug: "/docs/tutorial2"
date: "2020-09-03"
id: "tutorial2md"
--->""",
    3: """<!---
title: "Tutorial 3"
metaTitle: "Build a QA System Without Elasticsearch"
metaDescription: ""
slug: "/docs/tutorial3"
date: "2020-09-03"
id: "tutorial3md"
--->""",
    4: """<!---
title: "Tutorial 4"
metaTitle: "Utilizing existing FAQs for Question Answering"
metaDescription: ""
slug: "/docs/tutorial4"
date: "2020-09-03"
id: "tutorial4md"
--->""",
    5: """<!---
title: "Tutorial 5"
metaTitle: "Evaluation of a QA System"
metaDescription: ""
slug: "/docs/tutorial5"
date: "2020-09-03"
id: "tutorial5md"
--->""",
    6: """<!---
title: "Tutorial 6"
metaTitle: "Better retrieval via Dense Passage Retrieval"
metaDescription: ""
slug: "/docs/tutorial6"
date: "2020-09-03"
id: "tutorial6md"
--->""",
    7: """<!---
title: "Tutorial 7"
metaTitle: "Generative QA with RAG"
metaDescription: ""
slug: "/docs/tutorial7"
date: "2020-11-12"
id: "tutorial7md"
--->""",
    8: """<!---
title: "Tutorial 8"
metaTitle: "Preprocessing"
metaDescription: ""
slug: "/docs/tutorial8"
date: "2021-01-08"
id: "tutorial8md"
--->""",
    9: """<!---
title: "Tutorial 9"
metaTitle: "Training a Dense Passage Retrieval model"
metaDescription: ""
slug: "/docs/tutorial9"
date: "2021-01-08"
id: "tutorial9md"
--->""",
    10: """<!---
title: "Tutorial 10"
metaTitle: "Knowledge Graph QA"
metaDescription: ""
slug: "/docs/tutorial10"
date: "2021-04-06"
id: "tutorial10md"
--->""",
    11: """<!---
title: "Tutorial 11"
metaTitle: "Pipelines"
metaDescription: ""
slug: "/docs/tutorial11"
date: "2021-04-06"
id: "tutorial11md"
--->""",
    12: """<!---
title: "Tutorial 12"
metaTitle: "Generative QA with LFQA"
metaDescription: ""
slug: "/docs/tutorial12"
date: "2021-04-06"
id: "tutorial12md"
--->""",
    13: """<!---
title: "Tutorial 13"
metaTitle: "Question Generation"
metaDescription: ""
slug: "/docs/tutorial13"
date: "2021-08-23"
id: "tutorial13md"
--->""",
    14: """<!---
title: "Tutorial 14"
metaTitle: "Query Classifier Tutorial"
metaDescription: ""
slug: "/docs/tutorial14"
date: "2021-08-23"
id: "tutorial14md"
--->""",
    15: """<!---
title: "Tutorial 15"
metaTitle: "TableQA Tutorial"
metaDescription: ""
slug: "/docs/tutorial15"
date: "2021-10-28"
id: "tutorial15md"
--->""",
    16: """<!---
title: "Tutorial 16"
metaTitle: "DocumentClassifier at Index Time Tutorial"
metaDescription: ""
slug: "/docs/tutorial16"
date: "2021-11-05"
id: "tutorial16md"
--->""",
    17: """<!---
title: "Tutorial 17"
metaTitle: "Audio Tutorial"
metaDescription: ""
slug: "/docs/tutorial17"
date: "2022-06-15"
id: "tutorial17md"
--->""",
    18: """<!---
title: "Tutorial 18"
metaTitle: "GPL Domain Adaptation"
metaDescription: ""
slug: "/docs/tutorial18"
date: "2022-06-22"
id: "tutorial18md"
--->""",
}

notebook_tutorials_dir = Path(__file__).parent.parent.parent / "tutorials"
markdown_tutorials_dir = Path(__file__).parent.parent.parent / "docs" / "_src" / "tutorials" / "tutorials"
md_exporter = MarkdownExporter(exclude_output=True)


def get_notebook_number(nb_path: str):
    return int(re.search("\d+", nb_path.split("_")[0]).group(0))


def generate_markdown_from_notebook(nb_path: str):
    body, _ = md_exporter.from_filename(nb_path)
    n = get_notebook_number(nb_path)
    print(f"Processing {nb_path}")

    with open(markdown_tutorials_dir / f"{n}.md", "w", encoding="utf-8") as f:
        try:
            f.write(headers[n] + "\n\n")
        except IndexError as err:
            raise IndexError(
                "Can't find the header for this tutorial. Have you added it in '.github/utils/convert_notebooks_into_webpages.py'?"
            )
        f.write(body)


def main(argv: Sequence[str] = sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*", help="Filenames to check.")
    args = parser.parse_args(argv)

    for filename in args.filenames:
        filepath = pathlib.Path(filename)
        if filepath.parent == notebook_tutorials_dir and filepath.suffix == ".ipynb":
            generate_markdown_from_notebook(str(filepath))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
