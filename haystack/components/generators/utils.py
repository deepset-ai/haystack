from haystack.dataclasses import StreamingChunk


def print_streaming_chunk(chunk: StreamingChunk) -> None:
    """
    Default callback function for streaming responses.
    Prints the tokens of the first completion to stdout as soon as they are received
    """
    print(chunk.content, flush=True, end="")
