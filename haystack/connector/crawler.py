import sys
import json
import logging
import re
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Any, Optional, Dict, Tuple, Union
from haystack.schema import Document, BaseComponent
logger = logging.getLogger(__name__)


class Crawler(BaseComponent):
    """
    Crawl texts from a website so that we can use them later in Haystack as a corpus for search / question answering etc.

    **Example:**
    ```python
    |    from haystack.connector import Crawler
    |
    |    crawler = Crawler(output_dir="crawled_files")
    |    # crawl Haystack docs, i.e. all pages that include haystack.deepset.ai/overview/
    |    docs = crawler.crawl(urls=["https://haystack.deepset.ai/overview/get-started"],
    |                         filter_urls= ["haystack\.deepset\.ai\/docs\/"])
    ```
    """

    outgoing_edges = 1

    def __init__(self, output_dir: str, urls: Optional[List[str]] = None, crawler_depth: int = 1,
                 filter_urls: Optional[List] = None, overwrite_existing_files=True):
        """
        Init object with basic params for crawling (can be overwritten later).

        :param output_dir: Path for the directory to store files
        :param urls: List of http(s) address(es) (can also be supplied later when calling crawl())
        :param crawler_depth: How many sublinks to follow from the initial list of URLs. Current options:
                              0: Only initial list of urls
                              1: Follow links found on the initial URLs (but no further)
        :param filter_urls: Optional list of regular expressions that the crawled URLs must comply with.
                           All URLs not matching at least one of the regular expressions will be dropped.
        :param overwrite_existing_files: Whether to overwrite existing files in output_dir with new content
        """
        IN_COLAB = "google.colab" in sys.modules
        
        try:
            from webdriver_manager.chrome import ChromeDriverManager
        except ImportError:
            raise ImportError("Can't find package `webdriver-manager` \n"
                              "You can install it via `pip install webdriver-manager`")

        try:
            from selenium import webdriver
        except ImportError:
            raise ImportError("Can't find package `selenium` \n"
                              "You can install it via `pip install selenium`")

        options = webdriver.chrome.options.Options()
        options.add_argument('--headless')
        if IN_COLAB:
            try:
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                self.driver = webdriver.Chrome('chromedriver', options=options)
            except :
                raise Exception(
        """
        \'chromium-driver\' needs to be installed manually when running colab. Follow the below given commands:
                        !apt-get update
                        !apt install chromium-driver
                        !cp /usr/lib/chromium-browser/chromedriver /usr/bin
        If it has already been installed, please check if it has been copied to the right directory i.e. to \'/usr/bin\'"""
        )
        else:
            logger.info("'chrome-driver' will be automatically installed.")
            self.driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
        self.urls = urls
        self.output_dir = output_dir
        self.crawler_depth = crawler_depth
        self.filter_urls = filter_urls
        self.overwrite_existing_files = overwrite_existing_files

    def crawl(self,  output_dir: Union[str, Path, None] = None,
              urls: Optional[List[str]] = None,
              crawler_depth: Optional[int] = None,
              filter_urls: Optional[List] = None,
              overwrite_existing_files: Optional[bool] = None) -> List[Path]:
        """
        Craw URL(s), extract the text from the HTML, create a Haystack Document object out of it and save it (one JSON
        file per URL, including text and basic meta data).
        You can optionally specify via `filter_urls` to only crawl URLs that match a certain pattern.
        All parameters are optional here and only meant to overwrite instance attributes at runtime.
        If no parameters are provided to this method, the instance attributes that were passed during __init__ will be used.

        :param output_dir: Path for the directory to store files
        :param urls: List of http addresses or single http address
        :param crawler_depth: How many sublinks to follow from the initial list of URLs. Current options:
                              0: Only initial list of urls
                              1: Follow links found on the initial URLs (but no further)
        :param filter_urls: Optional list of regular expressions that the crawled URLs must comply with.
                           All URLs not matching at least one of the regular expressions will be dropped.
        :param overwrite_existing_files: Whether to overwrite existing files in output_dir with new content

        :return: List of paths where the crawled webpages got stored
        """
        # use passed params or fallback to instance attributes
        urls = urls or self.urls
        if urls is None:
            raise ValueError("Got no urls to crawl. Set `urls` to a list of URLs in __init__(), crawl() or run(). `")
        output_dir = output_dir or self.output_dir
        filter_urls = filter_urls or self.filter_urls
        if overwrite_existing_files is None:
            overwrite_existing_files = self.overwrite_existing_files
        if crawler_depth is None:
            crawler_depth = self.crawler_depth

        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        file_paths: list = []
        is_not_empty = len(list(output_dir.rglob("*"))) > 0
        if is_not_empty and not overwrite_existing_files:
            logger.info(
                f"Found data stored in `{output_dir}`. Delete this first if you really want to fetch new data."
            )
        else:
            logger.info(f"Fetching from {urls} to `{output_dir}`")
            sub_links: Dict[str, List] = {}

            # don't go beyond the initial list of urls
            if crawler_depth == 0:
                file_paths += self._write_to_files(urls, output_dir=output_dir)
            # follow one level of sublinks
            elif crawler_depth == 1:
                for url_ in urls:
                    existed_links: List = list(sum(list(sub_links.values()), []))
                    sub_links[url_] = list(self._extract_sublinks_from_url(base_url=url_, filter_urls=filter_urls,
                                                                     existed_links=existed_links))
                for url in sub_links:
                    file_paths += self._write_to_files(sub_links[url], output_dir=output_dir, base_url=url)

        return file_paths

    def _write_to_files(self, urls: List[str], output_dir: Path, base_url: str = None) -> List[Path]:
        paths = []
        for link in urls:
            logger.info(f"writing contents from `{link}`")
            self.driver.get(link)
            el = self.driver.find_element_by_tag_name('body')
            text = el.text

            link_split_values = link.replace('https://', '').split('/')
            file_name = f"{'_'.join(link_split_values)}.json"
            file_path = output_dir / file_name

            data = {}
            data['meta'] = {'url': link}
            if base_url:
                data['meta']['base_url'] = base_url
            data['text'] = text
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            paths.append(file_path)

        return paths

    def run(  # type: ignore
            self, 
            output_dir: Union[str, Path, None] = None,
            urls: Optional[List[str]] = None,
            crawler_depth: Optional[int] = None,
            filter_urls: Optional[List] = None,
            overwrite_existing_files: Optional[bool] = None,
            return_documents: Optional[bool] = False,
    ) -> Tuple[Dict, str]:
        """
        Method to be executed when the Crawler is used as a Node within a Haystack pipeline.

        :param output_dir: Path for the directory to store files
        :param urls: List of http addresses or single http address
        :param crawler_depth: How many sublinks to follow from the initial list of URLs. Current options:
                              0: Only initial list of urls
                              1: Follow links found on the initial URLs (but no further)
        :param filter_urls: Optional list of regular expressions that the crawled URLs must comply with.
                           All URLs not matching at least one of the regular expressions will be dropped.
        :param overwrite_existing_files: Whether to overwrite existing files in output_dir with new content
        :param return_documents:  Return json files content

        :return: Tuple({"paths": List of filepaths, ...}, Name of output edge)
        """

        file_paths = self.crawl(urls=urls, output_dir=output_dir, crawler_depth=crawler_depth,
                                  filter_urls=filter_urls, overwrite_existing_files=overwrite_existing_files)
        if return_documents:
            crawled_data = [self._read_json_file(_file) for _file in file_paths]
            results = {"documents": crawled_data}
        else:
            results = {"paths": file_paths}

        return results, "output_1"

    @staticmethod
    def _is_internal_url(base_url: str, sub_link: str) -> bool:
        base_url_ = urlparse(base_url)
        sub_link_ = urlparse(sub_link)
        return base_url_.scheme == sub_link_.scheme and base_url_.netloc == sub_link_.netloc

    @staticmethod
    def _is_inpage_navigation(base_url: str, sub_link: str) -> bool:
        base_url_ = urlparse(base_url)
        sub_link_ = urlparse(sub_link)
        return base_url_.path == sub_link_.path and base_url_.netloc == sub_link_.netloc

    def _extract_sublinks_from_url(self, base_url: str,
                                   filter_urls: Optional[List] = None,
                                  existed_links: List = None) -> set:
        self.driver.get(base_url)
        a_elements = self.driver.find_elements_by_tag_name('a')
        sub_links = set()
        if not (existed_links and base_url in existed_links):
            if filter_urls:
                if re.compile('|'.join(filter_urls)).search(base_url):
                    sub_links.add(base_url)

        for i in a_elements:
            sub_link = i.get_attribute('href')
            if not (existed_links and sub_link in existed_links):
                if self._is_internal_url(base_url=base_url, sub_link=sub_link) \
                        and (not self._is_inpage_navigation(base_url=base_url, sub_link=sub_link)):
                    if filter_urls:
                        if re.compile('|'.join(filter_urls)).search(sub_link):
                            sub_links.add(sub_link)
                    else:
                        sub_links.add(sub_link)

        return sub_links

    def _read_json_file(self, file_path: Path):
        """Read the json file and return the content"""
        with open(file_path.absolute(), "r") as read_file:
            return json.load(read_file)
