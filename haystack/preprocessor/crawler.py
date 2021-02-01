import logging
import re
from pathlib import Path
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urlparse
from typing import List, Any, Optional, Dict

logger = logging.getLogger(__name__)



def fetch_data_from_urls(urls: Any, output_dir: str, extract_sub_links: bool = True, include: Optional[str] = None):
    """
    takes url/urls as input and write contents to files

    :param urls: list of http addresses or single http address
    :type urls: str or list (Any)
    :param output_dir: path for directory to store files
    :type output_dir: str
    :param extract_sub_links: whether to extract sub-links from urls or not
    :type extract_sub_links: bool
    :param include: regex to include matching urls only
    :type include: optional
    :return : None
    """

    path = Path(output_dir)
    if not path.exists():
        path.mkdir(parents=True)
        
    is_not_empty = len(list(Path(path).rglob("*"))) > 0
    if is_not_empty:
        logger.info(
            f"Found data stored in `{output_dir}`. Delete this first if you really want to fetch new data."
        )
        return False
    else:   
        logger.info(f"Fetching from {urls} to `{output_dir}`")
        options = webdriver.chrome.options.Options()
        options.add_argument('--headless')
        driver = webdriver.Chrome(ChromeDriverManager().install(),  options=options)

        docs = []

        if type(urls) != list:
            urls = [urls]

        sub_links: Dict[str, List] = {}
        if extract_sub_links==True:
            for url_ in urls:
                existed_links: List = list(sum(list(sub_links.values()), []))
                sub_links[url_] = list(extract_sublinks_from_url(base_url=url_, driver=driver, include=include, existed_links=existed_links))

            for url in sub_links:
                docs += write_to_files(sub_links[url], driver=driver, output_dir=output_dir, base_url=url)
        else:
            for url_ in urls:
                docs += write_to_files(url_, driver=driver, output_dir=output_dir)
        
        return docs


def write_to_files(urls: Any, driver: Any, output_dir: str, base_url: str = None):
    if type(urls) != list:
        urls = [urls]
    
    docs = []

    for link in urls:
        logger.info(f"writing contents from `{link}`")
        driver.get(link)
        el = driver.find_element_by_tag_name('body')
        text = el.text

        link_split_values = link.replace('https://','').split('/')
        file_name = '{}/{}.txt'.format(output_dir, '_'.join(link_split_values))

        data = {}
        data['meta'] = {'url': link}
        if base_url:
            data['meta']['base_url'] = base_url
        data['text'] = text
        with open(file_name, 'w') as f:
            f.write(str(data))
        
        docs.append(data)
    
    return docs


def is_internal_url(base_url: str, sub_link: str):
    base_url_ = urlparse(base_url)
    sub_link_ = urlparse(sub_link)

    return base_url_.scheme == sub_link_.scheme and base_url_.netloc == sub_link_.netloc


def is_inpage_navigation(base_url: str, sub_link: str):
    base_url_ = urlparse(base_url)
    sub_link_ = urlparse(sub_link)

    return base_url_.path == sub_link_.path and base_url_.netloc == sub_link_.netloc


def extract_sublinks_from_url(base_url: str, driver: Any, include: Optional[str] = None, existed_links: List = None):
    driver.get(base_url)
    a_elements = driver.find_elements_by_tag_name('a')
    sub_links = set()
    if not (existed_links and base_url in existed_links):
        if include:
            if re.compile(include).search(base_url):
                sub_links.add(base_url)

    for i in a_elements:
        sub_link = i.get_attribute('href')
        if not (existed_links and sub_link in existed_links):
            if is_internal_url(base_url=base_url, sub_link=sub_link) \
                and (not is_inpage_navigation(base_url=base_url, sub_link=sub_link)):
                if include:
                    if re.compile(include).search(sub_link):
                        sub_links.add(sub_link)
                else:
                    sub_links.add(sub_link)

    return sub_links


    