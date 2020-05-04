import tarfile
import time
import urllib.request
from subprocess import Popen, PIPE, STDOUT

import pytest


@pytest.fixture(scope="session")
def elasticsearch_fixture(tmpdir):
    thetarfile = "https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.6.1-linux-x86_64.tar.gz"
    ftpstream = urllib.request.urlopen(thetarfile)
    thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
    thetarfile.extractall(path=tmpdir)
    es_server = Popen(["elasticsearch-7.6.1/bin/elasticsearch"], stdout=PIPE, stderr=STDOUT)
    time.sleep(20)
