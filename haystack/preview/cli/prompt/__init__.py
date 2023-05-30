import click

from haystack.preview.cli.prompt import cache


@click.group(short_help="Prompts related commands")
def prompt():
    pass


prompt.add_command(cache.cache)
