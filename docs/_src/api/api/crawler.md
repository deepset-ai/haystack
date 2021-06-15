<a name="crawler"></a>
# Module crawler

<a name="crawler.Crawler"></a>
## Crawler Objects

```python
class Crawler(BaseComponent)
```

Crawl texts from a website so that we can use them later in Haystack as a corpus for search / question answering etc.

**Example:**
```python
|    from haystack.connector import Crawler
|
|    crawler = Crawler()
|    # crawl Haystack docs, i.e. all pages that include haystack.deepset.ai/docs/
|    docs = crawler.crawl(urls=["https://haystack.deepset.ai/docs/latest/get_startedmd"],
|                         output_dir="crawled_files",
|                         filter_urls= ["haystack\.deepset\.ai\/docs\/"])
```

<a name="crawler.Crawler.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(output_dir: str, urls: Optional[List[str]] = None, crawler_depth: int = 1, filter_urls: Optional[List] = None, overwrite_existing_files=True)
```

Init object with basic params for crawling (can be overwritten later).

**Arguments**:

- `output_dir`: Path for the directory to store files
- `urls`: List of http(s) address(es) (can also be supplied later when calling crawl())
- `crawler_depth`: How many sublinks to follow from the initial list of URLs. Current options:
                      0: Only initial list of urls
                      1: Follow links found on the initial URLs (but no further)
- `filter_urls`: Optional list of regular expressions that the crawled URLs must comply with.
                   All URLs not matching at least one of the regular expressions will be dropped.
- `overwrite_existing_files`: Whether to overwrite existing files in output_dir with new content

<a name="crawler.Crawler.crawl"></a>
#### crawl

```python
 | crawl(output_dir: Union[str, Path, None] = None, urls: Optional[List[str]] = None, crawler_depth: Optional[int] = None, filter_urls: Optional[List] = None, overwrite_existing_files: Optional[bool] = None) -> List[Path]
```

Craw URL(s), extract the text from the HTML, create a Haystack Document object out of it and save it (one JSON
file per URL, including text and basic meta data).
You can optionally specify via `filter_urls` to only crawl URLs that match a certain pattern.
All parameters are optional here and only meant to overwrite instance attributes at runtime.
If no parameters are provided to this method, the instance attributes that were passed during __init__ will be used.

**Arguments**:

- `output_dir`: Path for the directory to store files
- `urls`: List of http addresses or single http address
- `crawler_depth`: How many sublinks to follow from the initial list of URLs. Current options:
                      0: Only initial list of urls
                      1: Follow links found on the initial URLs (but no further)
- `filter_urls`: Optional list of regular expressions that the crawled URLs must comply with.
                   All URLs not matching at least one of the regular expressions will be dropped.
- `overwrite_existing_files`: Whether to overwrite existing files in output_dir with new content

**Returns**:

List of paths where the crawled webpages got stored

<a name="crawler.Crawler.run"></a>
#### run

```python
 | run(output_dir: Union[str, Path, None] = None, urls: Optional[List[str]] = None, crawler_depth: Optional[int] = None, filter_urls: Optional[List] = None, overwrite_existing_files: Optional[bool] = None, **kwargs) -> Tuple[Dict, str]
```

Method to be executed when the Crawler is used as a Node within a Haystack pipeline.

**Arguments**:

- `output_dir`: Path for the directory to store files
- `urls`: List of http addresses or single http address
- `crawler_depth`: How many sublinks to follow from the initial list of URLs. Current options:
                      0: Only initial list of urls
                      1: Follow links found on the initial URLs (but no further)
- `filter_urls`: Optional list of regular expressions that the crawled URLs must comply with.
                   All URLs not matching at least one of the regular expressions will be dropped.
- `overwrite_existing_files`: Whether to overwrite existing files in output_dir with new content

**Returns**:

Tuple({"paths": List of filepaths, ...}, Name of output edge)

