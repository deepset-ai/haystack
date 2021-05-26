<!---
title: "Preprocessing"
metaTitle: "Preprocessing"
metaDescription: ""
slug: "/docs/preprocessing"
date: "2020-09-03"
id: "preprocessingmd"
--->

# Preprocessing

Haystack includes a suite of tools to:
 
* extract text from different file types, 
* normalize white space
* split text into smaller pieces to optimize retrieval

<div class="recommendation">

Check out our [preprocessing tutorial](/docs/latest/tutorial8md) if you'd like to start working with code examples already!

</div>

These data preprocessing steps can have a big impact on the systems performance
and effective handling of data is key to getting the most out of Haystack.

The Document Store expects its inputs to come in the following format. 
The sections below will show you all the tools you'll need to ready your data for storing.
 
```python
docs = [
    {
        'text': DOCUMENT_TEXT_HERE,
        'meta': {'name': DOCUMENT_NAME, ...}
    }, ...
]
```

## File Conversion

There are a range of different file converters in Haystack that 
can extract text from files and cast them into the unified dictionary format shown above.
Haystack features support for txt, pdf and docx files and there is even a converter that leverages Apache Tika.
Please refer to [the API docs](/docs/latest/file_convertersmd) to see which converter best suits you.

<div class="tabs tabsconverters">

<div class="tab">
<input type="radio" id="tab-1" name="tab-group-1" checked>
<label class="labelouter" for="tab-1">PDF</label>
<div class="tabcontent">

```python
from haystack.file_converter import PDFToTextConverter

converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["de","en"])
doc = converter.convert(file_path=file, meta=None)
```

</div>
</div>

<div class="tab">
<input type="radio" id="tab-2" name="tab-group-1">
<label class="labelouter" for="tab-2">DOCX</label>
<div class="tabcontent">

```python
from haystack.file_converter import DocxToTextConverter

converter = DocxToTextConverter(remove_numeric_tables=True, valid_languages=["de","en"])
doc = converter.convert(file_path=file, meta=None)
```

</div>
</div>

<div class="tab">
<input type="radio" id="tab-3" name="tab-group-1">
<label class="labelouter" for="tab-3">From a Directory</label>
<div class="tabcontent">


Haystack also has a `convert_files_to_dicts()` utility function that will convert
all txt or pdf files in a given folder into this dictionary format.

```python
from haystack.preprocessor.utils import convert_files_to_dicts

docs = convert_files_to_dicts(dir_path=doc_dir)
```

</div>
</div>

</div>

## Web Crawler

In Haystack, you will find a web crawler that will help you scrape text from websites and save it to file. 
See the [API documentation](https://haystack.deepset.ai/docs/latest/apicrawlermd) for more details.

```python
from haystack.connector import Crawler

crawler = Crawler()
docs = crawler.crawl(urls=["https://haystack.deepset.ai/docs/latest/get_startedmd"],
                     output_dir="crawled_files",
                     filter_urls= ["haystack\.deepset\.ai\/docs\/"])
```

## PreProcessor

While each of the above conversion methods produce documents that are already in the format expected by the Document Store,
it is recommended that they are further processed in order to ensure optimal Retriever and Reader performance.
The `PreProcessor` takes one of the documents created by the converter as input,
performs various cleaning steps and splits them into multiple smaller documents.

For suggestions on how best to split your documents, see [Optimization](/docs/latest/optimizationmd)

```python
from haystack.preprocessor import PreProcessor

doc = converter.convert(file_path=file, meta=None)
processor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="word",
    split_length=200,
    split_respect_sentence_boundary=True,
    split_overlap=0
)
docs = processor.process(doc)
```

* `clean_empty_lines` will normalize 3 or more consecutive empty lines to be just a two empty lines
* `clean_whitespace` will remove any whitespace at the beginning or end of each line in the text
* `clean_header_footer` will remove any long header or footer texts that are repeated on each page
* `split_by` determines what unit the document is split by: `'word'`, `'sentence'` or `'passage'`
* `split_length` sets a maximum number of `'word'`, `'sentence'` or `'passage'` units per output document
* `split_respect_sentence_boundary` ensures that document boundaries do not fall in the middle of sentences
* `split_overlap` sets the amount of overlap between two adjacent documents after a split. Setting this to a positive number essentially enables the sliding window approach.

