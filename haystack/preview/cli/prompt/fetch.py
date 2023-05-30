import click

from haystack.nodes.prompt.prompt_template import PromptNotFoundError, fetch_from_prompthub, cache_prompt


@click.command(
    short_help="Downloads and saves prompts from Haystack PromptHub",
    help="""
    Downloads a prompt from the official Haystack PromptHub and saves
    it locally to ease use in enviroments with no network.

    PROMPT_NAME can be specified multiple times.

    PROMPTHUB_CACHE_PATH environment variable can be set to change the
    default folder in which the prompts will be saved in.

    If a custom PROMPTHUB_CACHE_PATH is used remember to also used it
    for Haystack invocations.

    The Haystack PromptHub is https://prompthub.deepset.ai/
    """,
)
@click.argument("prompt_name", nargs=-1)
def fetch(prompt_name):
    for name in prompt_name:
        try:
            data = fetch_from_prompthub(name)
        except PromptNotFoundError as err:
            raise click.ClickException(str(err)) from err
        cache_prompt(data)
