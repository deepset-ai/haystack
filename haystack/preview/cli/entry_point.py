import click

from haystack import __version__
from haystack.preview.cli.rest_api import rest_api, serve


@click.group()
@click.version_option(__version__)
@click.pass_context
def main_cli(ctx):
    pass


def main():
    main_cli()
    rest_api.add_command(serve)
    rest_api()
