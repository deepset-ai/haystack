import click

from haystack import __version__
from haystack.cli.prompt import prompt


@click.group()
@click.version_option(__version__)
def main_cli():
    pass


main_cli.add_command(prompt)


def main():
    main_cli()
