import click

from haystack.nodes.prompt.prompt_template import PromptNotFoundError, fetch_from_prompthub, cache_prompt


@click.command(
    short_help="Downloads and saves prompts from Haystack PromptHub",
    help="""
    Downloads a prompt from the official [Haystack PromptHub](https://prompthub.deepset.ai/)
    and saves it locally for easier use in environments with no network.

    You can specify multiple prompts to fetch at the same time.

    PROMPTHUB_CACHE_PATH environment variable can be set to change the default
    folder in which the prompts will be saved in. You can find the default cache path on your machine by running the following code:

    ```
    from haystack.nodes.prompt.prompt_template import PROMPTHUB_CACHE_PATH
    print(PROMPTHUB_CACHE_PATH)
    ```

    If you set a custom PROMPTHUB_CACHE_PATH environment variable, remember to
    set it to the same value in your console before running Haystack.
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
