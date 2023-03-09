from dataclasses import dataclass


@dataclass
class Span:
    start: int
    end: int
    """
    Defining a sequence of characters (Text span) or cells (Table span) via start and end index.
    For extractive QA: Character where answer starts/ends
    For TableQA: Cell where the answer starts/ends (counted from top left to bottom right of table)

    :param start: Position where the span starts
    :param end:  Position where the spand ends
    """

    def __contains__(self, value):
        """
        Checks for inclusion of the given value into the interval defined by Span.
        ```
            assert 10 in Span(5, 15)  # True
            assert 20 in Span(1, 15)  # False
        ```
        Includes the left edge, but not the right edge.
        ```
            assert 5 in Span(5, 15)   # True
            assert 15 in Span(5, 15)  # False
        ```
        Works for numbers and all values that can be safely converted into floats.
        ```
            assert 10.0 in Span(5, 15)   # True
            assert "10" in Span(5, 15)   # True
        ```
        It also works for Span objects, returning True only if the given
        Span is fully contained into the original Span.
        As for numerical values, the left edge is included, the right edge is not.
        ```
            assert Span(10, 11) in Span(5, 15)   # True
            assert Span(5, 10) in Span(5, 15)    # True
            assert Span(10, 15) in Span(5, 15)   # False
            assert Span(5, 15) in Span(5, 15)    # False
            assert Span(5, 14) in Span(5, 15)    # True
            assert Span(0, 1) in Span(5, 15)     # False
            assert Span(0, 10) in Span(5, 15)    # False
            assert Span(10, 20) in Span(5, 15)   # False
        ```
        """
        if isinstance(value, Span):
            return self.start <= value.start and self.end > value.end
        try:
            value = float(value)
            return self.start <= value < self.end
        except Exception as e:
            raise ValueError(
                f"Cannot use 'in' with a value of type {type(value)}. Use numeric values or Span objects."
            ) from e
