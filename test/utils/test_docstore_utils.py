import subprocess
from unittest.mock import patch

import pytest


from haystack.utils import launch_es


@pytest.mark.unit
@patch("haystack.utils.doc_store.subprocess.run")
def test_launch_es_default_values(subprocess_run_mock):
    expected_docker_command = (
        "docker start elasticsearch > /dev/null 2>&1 || docker run -d -p 9200:9200 -p 9600:9600 -e "
        '"discovery.type=single-node"  -e "xpack.security.enabled=false" --name elasticsearch elasticsearch:7.17.6'
    )
    launch_es()
    subprocess_run_mock.assert_called_once_with([expected_docker_command], shell=True)


@pytest.mark.unit
@patch("haystack.utils.doc_store.subprocess.run")
def test_launch_es_delete_existing(subprocess_run_mock):
    expected_docker_command = "docker rm --force elasticsearch"
    launch_es(delete_existing=True)
    subprocess_run_mock.assert_any_call([expected_docker_command], shell=True, stdout=subprocess.DEVNULL)
    assert subprocess_run_mock.call_count == 2


@pytest.mark.unit
@patch("haystack.utils.doc_store.subprocess.run")
def test_launch_es_custom_version_tag(subprocess_run_mock):
    expected_docker_command = (
        "docker start elasticsearch > /dev/null 2>&1 || docker run -d -p 9200:9200 -p 9600:9600 -e "
        '"discovery.type=single-node"  -e "xpack.security.enabled=false" --name elasticsearch elasticsearch:CUSTOM_TAG'
    )
    launch_es(version_tag="CUSTOM_TAG")
    subprocess_run_mock.assert_called_once_with([expected_docker_command], shell=True)


@pytest.mark.unit
@patch("haystack.utils.doc_store.subprocess.run")
def test_launch_es_with_password(subprocess_run_mock):
    expected_docker_command = (
        "docker start elasticsearch > /dev/null 2>&1 || docker run -d -p 9200:9200 -p 9600:9600 -e "
        '"discovery.type=single-node"  -e "ELASTIC_PASSWORD=PASSWORD" -e "xpack.security.enabled=true" '
        "--name elasticsearch elasticsearch:7.17.6"
    )
    launch_es(password="PASSWORD")
    subprocess_run_mock.assert_called_once_with([expected_docker_command], shell=True)


@pytest.mark.unit
@patch("haystack.utils.doc_store.subprocess.run")
def test_launch_es_with_java_opts(subprocess_run_mock):
    expected_docker_command = (
        "docker start elasticsearch > /dev/null 2>&1 || docker run -d -p 9200:9200 -p 9600:9600 -e "
        '"discovery.type=single-node" -e ES_JAVA_OPTS="CUSTOM_OPTS"  -e "xpack.security.enabled=false" --name '
        "elasticsearch elasticsearch:7.17.6"
    )
    launch_es(java_opts="CUSTOM_OPTS")
    subprocess_run_mock.assert_called_once_with([expected_docker_command], shell=True)
