import click

from haystack import __version__


@click.group()
@click.version_option(__version__)
def main_cli():
    pass


def main():
    main_cli()
