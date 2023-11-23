from itertools import islice


def get_batches_from_generator(iterable, n):
    """
    Batch elements of an iterable into fixed-length chunks or blocks.
    """
    it = iter(iterable)
    x = tuple(islice(it, n))
    while x:
        yield x
        x = tuple(islice(it, n))
