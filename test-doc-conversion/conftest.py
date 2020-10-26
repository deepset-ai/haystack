import subprocess
import time
from subprocess import run
from sys import platform

import pytest
import requests


@pytest.fixture(scope="session")
def tika_fixture():
    try:
        tika_url = "http://localhost:9998/tika"
        ping = requests.get(tika_url)
        if ping.status_code != 200:
            raise Exception(
                "Unable to connect Tika. Please check tika endpoint {0}.".format(tika_url))
    except:
        print("Starting Tika ...")
        status = subprocess.run(
            ['docker run -d --name tika -p 9998:9998 apache/tika:1.24.1'],
            shell=True
        )
        if status.returncode:
            raise Exception(
                "Failed to launch Tika. Please check docker container logs.")
        time.sleep(30)


@pytest.fixture(scope="session")
def xpdf_fixture(tika_fixture):
    verify_installation = run(["pdftotext"], shell=True)
    if verify_installation.returncode == 127:
        if platform.startswith("linux"):
            platform_id = "linux"
            sudo_prefix = "sudo"
        elif platform.startswith("darwin"):
            platform_id = "mac"
            # For Mac, generally sudo need password in interactive console.
            # But most of the cases current user already have permission to copy to /user/local/bin.
            # Hence removing sudo requirement for Mac.
            sudo_prefix = ""
        else:
            raise Exception(
                """Currently auto installation of pdftotext is not supported on {0} platform """.format(platform)
            )

        commands = """ wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-{0}-4.02.tar.gz &&
                       tar -xvf xpdf-tools-{0}-4.02.tar.gz &&
                       {1} cp xpdf-tools-{0}-4.02/bin64/pdftotext /usr/local/bin""".format(platform_id, sudo_prefix)
        run([commands], shell=True)

        verify_installation = run(["pdftotext -v"], shell=True)
        if verify_installation.returncode == 127:
            raise Exception(
                """pdftotext is not installed. It is part of xpdf or poppler-utils software suite.
                 You can download for your OS from here: https://www.xpdfreader.com/download.html."""
            )
