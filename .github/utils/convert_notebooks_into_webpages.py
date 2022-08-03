#!/usr/bin/env python3

import re

from nbconvert import MarkdownExporter
import os
from pathlib import Path

headers = {
    1: """---
title: Build Your First QA System
excerpt: Haystack Tutorial
category: 62ea2cd68129310085ac8d37
slug: build-your-first-qa-system
order: 0
hidden: false
---""",
    2: """---
title: Fine-tuning a Model on Your Own Data
excerpt: Haystack Tutorial
category: 62ea2cd68129310085ac8d37
slug: fine-tuning-a-model-on-your-own-data
order: 10
hidden: false
---""",
    3: """---
title: Build a QA System Without Elasticsearch
excerpt: Haystack Tutorial
category: 62ea2cd68129310085ac8d37
slug: build-a-qa-system-without-elasticsearch
order: 20
hidden: false
---""",
    4: """---
title: Utilizing existing FAQs for Question Answering
excerpt: Haystack Tutorial
category: 62ea2cd68129310085ac8d37
slug: build-a-qa-system-without-elasticsearch
order: 30
hidden: false
---""",
    5: """---
title: Evaluation of a Pipeline and its Components
excerpt: Haystack Tutorial
category: 62ea2cd68129310085ac8d37
slug: evaluation-of-a-pipeline-and-its-components
order: 40
hidden: false
---""",
    6: """---
title: Better Retrieval via "Dense Passage Retrieval"
excerpt: Haystack Tutorial
category: 62ea2cd68129310085ac8d37
slug: better-retrieval-via-dense-passage-retrieval
order: 50
hidden: false
---""",
    7: """---
title: Generative QA with "Retrieval-Augmented Generation"
excerpt: Haystack Tutorial
category: 62ea2cd68129310085ac8d37
slug: generative-qa-with-retrieval-augmented-generation
order: 60
hidden: false
---""",
    8: """---
title: How to Preprocess Documents
excerpt: Haystack Tutorial
category: 62ea2cd68129310085ac8d37
slug: how-to-preprocess-documents
order: 70
hidden: false
---""",
    9: """---
title: Training a Dense Passage Retrieval model
excerpt: Haystack Tutorial
category: 62ea2cd68129310085ac8d37
slug: training-a-dense-passage-retrieval-model
order: 80
hidden: false
---""",
    10: """---
title: Knowledge Graph QA
excerpt: Haystack Tutorial
category: 62ea2cd68129310085ac8d37
slug: knowledge-graph-qa
order: 90
hidden: false
---""",
    11: """---
title: Working with Pipelines
excerpt: Haystack Tutorial
category: 62ea2cd68129310085ac8d37
slug: working-with-pipelines
order: 100
hidden: false
---""",
    12: """---
title: Generative QA with LFQA
excerpt: Haystack Tutorial
category: 62ea2cd68129310085ac8d37
slug: generative-qa-with-lfqa
order: 110
hidden: false
---""",
    13: """---
title: Question Generation
excerpt: Haystack Tutorial
category: 62ea2cd68129310085ac8d37
slug: question-generation
order: 120
hidden: false
---""",
    14: """---
title: Query Classifier Tutorial
excerpt: Haystack Tutorial
category: 62ea2cd68129310085ac8d37
slug: query-classifier-tutorial
order: 130
hidden: false
---""",
    15: """---
title: TableQA Tutorial
excerpt: Haystack Tutorial
category: 62ea2cd68129310085ac8d37
slug: tableqa-tutorial
order: 140
hidden: false
---""",
    16: """---
title: DocumentClassifier at Index Time Tutorial
excerpt: Haystack Tutorial
category: 62ea2cd68129310085ac8d37
slug: documentclassifier-at-index-time-tutorial
order: 150
hidden: false
---""",
    17: """---
title: Audio Tutorial
excerpt: Haystack Tutorial
category: 62ea2cd68129310085ac8d37
slug: audio-tutorial
order: 160
hidden: false
---""",
    18: """---
title: GPL Domain Adaptation
excerpt: Haystack Tutorial
category: 62ea2cd68129310085ac8d37
slug: gpl-domain-adaptation
order: 170
hidden: false
---"""
}

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    test = [atoi(c) for c in re.split("(\d+)", text)]
    return test


dir = Path(__file__).parent.parent.parent / "tutorials"

notebooks = [x for x in os.listdir(dir) if x[-6:] == ".ipynb"]
# sort notebooks based on numbers within name of notebook
notebooks = sorted(notebooks, key=lambda x: natural_keys(x))


e = MarkdownExporter(exclude_output=True)
for i, nb in enumerate(notebooks):
    body, resources = e.from_filename(dir / nb)
    # Remove title since it is provided by the header
    body = "\n".join(body.split("\n")[1:]).strip()
    print(f"Processing {dir}/{nb}")

    tutorials_path = Path(__file__).parent.parent.parent / "docs" / "_src" / "tutorials" / "tutorials"
    with open(tutorials_path / f"{i + 1}.md", "w", encoding="utf-8") as f:
        try:
            f.write(headers[i + 1] + "\n\n")
        except IndexError as e:
            raise IndexError(
                "Can't find the header for this tutorial. Have you added it in '.github/utils/convert_notebooks_into_webpages.py'?"
            )
        f.write(body)
