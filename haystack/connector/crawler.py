import logging
import re
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Any, Optional, Dict, Tuple
from haystack.schema import Document, BaseComponent
logger = logging.getLogger(__name__)


class Crawler(BaseComponent):
    """
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
    """

    outgoing_edges = 1

    def __init__(self):

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
        self.driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

    def crawl(self, urls: Any, output_dir: str, crawler_depth: int = 1, filter_urls: Optional[List] = None) -> List[Document]:
        """
        Craw URL(s), extract the text from the HTML, create a Haystack Document object out of it and save it (one JSON
        file per URL, including text and basic meta data).
        You can optionally specify via `filter_urls` to only crawl URLs that match a certain pattern.

        :param urls: List of http addresses or single http address
        :param output_dir: Path for the directory to store files
        :param crawler_depth: How many sublinks to follow from the initial list of URLs. Current options:
                              0: Only initial list of urls
                              1: Follow links found on the initial URLs (but no further)
        :param filter_urls: Optional list of regular expressions that the crawled URLs must comply with.
                           All URLs not matching at least one of the regular expressions will be dropped.
        :return: List of Documents
        """
        path = Path(output_dir)
        if not path.exists():
            path.mkdir(parents=True)

        is_not_empty = len(list(Path(path).rglob("*"))) > 0
        if is_not_empty :
            logger.info(
                f"Found data stored in `{output_dir}`. Delete this first if you really want to fetch new data."
            )
            return []
        else:
            logger.info(f"Fetching from {urls} to `{output_dir}`")

            docs = []
            if type(urls) != list:
                urls = [urls]

            sub_links: Dict[str, List] = {}

            # don't go beyond the initial list of urls
            if crawler_depth == 0:
                for url_ in urls:
                    docs += self._write_to_files(url_, output_dir=output_dir)
            # follow one level of sublinks
            elif crawler_depth == 1:
                for url_ in urls:
                    existed_links: List = list(sum(list(sub_links.values()), []))
                    sub_links[url_] = list(self._extract_sublinks_from_url(base_url=url_, filter_urls=filter_urls,
                                                                     existed_links=existed_links))
                for url in sub_links:
                    docs += self._write_to_files(sub_links[url], output_dir=output_dir, base_url=url)

            return docs

    def _write_to_files(self, urls: Any, output_dir: str, base_url: str = None):
        if type(urls) != list:
            urls = [urls]

        docs = []

        for link in urls:
            logger.info(f"writing contents from `{link}`")
            self.driver.get(link)
            el = self.driver.find_element_by_tag_name('body')
            text = el.text

            link_split_values = link.replace('https://', '').split('/')
            file_name = '{}/{}.json'.format(output_dir, '_'.join(link_split_values))

            data = {}
            data['meta'] = {'url': link}
            if base_url:
                data['meta']['base_url'] = base_url
            data['text'] = text
            with open(file_name, 'w') as f:
                f.write(str(data))

            new_doc = Document.from_dict(data)
            docs.append(new_doc)

        return docs

    def run(self, urls: Any,
            output_dir: str,
            crawler_depth: int = 1,
            filter_urls: Optional[List] = None) -> Tuple[List[Document], str]:
        """
        Method to be executed when the Crawler is used as a Node within a Haystack pipeline.

        :param urls: List of http addresses or single http address
        :param output_dir: Path for the directory to store files
        :param crawler_depth: How many sublinks to follow from the initial list of URLs. Current options:
                              0: Only initial list of urls
                              1: Follow links found on the initial URLs (but no further)
        :param filter_urls: Optional list of regular expressions that the crawled URLs must comply with.
                           All URLs not matching at least one of the regular expressions will be dropped.
        :return: (List of Documents, Name of output edge)
        """

        docs = self.crawl(urls=urls, output_dir=output_dir, crawler_depth=crawler_depth, filter_urls=filter_urls)
        return docs, "output_1"

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
