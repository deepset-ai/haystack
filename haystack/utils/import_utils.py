from typing import Optional

import io
import tarfile
import zipfile
import requests
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def fetch_archive_from_http(url: str, output_dir: str, proxies: Optional[dict] = None) -> bool:
    """
    Fetch an archive (zip or tar.gz) from a url via http and extract content to an output directory.

    :param url: http address
    :param output_dir: local path
    :param proxies: proxies details as required by requests library
    :return: if anything got fetched
    """
    # verify & prepare local directory
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
        logger.info(f"Fetching from {url} to `{output_dir}`")

        _, _, archive_extension = url.rpartition(".")
        request_data = requests.get(url, proxies=proxies)

        if archive_extension == "zip":
            zip_archive = zipfile.ZipFile(io.BytesIO(request_data.content))
            zip_archive.extractall(output_dir)
        elif archive_extension in ["gz", "bz2", "xz"]:
            tar_archive = tarfile.open(fileobj=io.BytesIO(request_data.content), mode="r|*")
            tar_archive.extractall(output_dir)
        else:
            logger.warning('Skipped url {0} as file type is not supported here. '
                           'See haystack documentation for support of more file types'.format(url))

        return True
