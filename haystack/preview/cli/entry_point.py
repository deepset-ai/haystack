import click

from haystack import __version__


@click.group()
@click.version_option(__version__)
@click.pass_context
def main_cli(ctx):
    pass


def main():
    main_cli()
