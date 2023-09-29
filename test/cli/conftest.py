import pytest
from click.testing import CliRunner


@pytest.fixture()
def cli_runner():
    yield CliRunner()
