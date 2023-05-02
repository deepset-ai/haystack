from typing import Optional, Any

from dataclasses import dataclass

from canals import component


class _MetaClass(type):
    def __call__(cls, *args, **kwargs):

        # The __call__ method of a metaclass is called early enough to make the validation of the actual class succeed.
        # This method is responsible of calling both __new__ and __init__ properly.

        expected_type = kwargs.pop("expected_type", None)
        if not expected_type:
            raise ValueError(
                "Please specify the type MergeLoop should expect by giving a value to the 'expected_type' parameter.\n"
                "For example: 'merge_loop = MergeLoop(expected_type=List[Document])'"
            )

        # Here is where the Output dataclass and the run method are defined with the specific types.
        # Let's minimize the amount of logic contained here to the absolute minimum, because this area of the code
        # is really complex to reason about.

        @dataclass
        class Output:
            value: Optional[expected_type]

        def run(
            self, first_branch: Optional[expected_type] = None, second_branch: Optional[expected_type] = None
        ) -> Output:
            return self._run(first_branch=first_branch, second_branch=second_branch)

        cls.Output = Output
        cls.run = run

        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        return obj


@component
class MergeLoop(metaclass=_MetaClass):
    """
    Takes two input components and returns the first one that is not None.

    In case both received a value, priority is given to 'first_branch'.

    Always initialize this class by passing the expected type, like: `MergeLoop(expected_type=int)`.
    """

    #
    # Note: highly dynamically typed classes like MergeLoop are mostly built by the metaclass.
    # However, @component's reflexive checks will fail if the class does not have a run() method
    # that returns an Output dataclass, so we have to provide these mocks to make it happy.
    #
    # The real Output and run() are both defined in the metaclass.__call__(). Note how run() in fact calls
    # back directly _run() to avoid encoding too much logic in the metaclass itself.
    #
    @dataclass
    class Output:
        pass

    def run(self, first_branch: Any, second_branch: Any) -> Output:  # type: ignore
        """
        Takes two input components and returns the first one that is not None.

        In case both received a value, priority is given to 'first_branch'.
        """
        pass

    def _run(self, first_branch: Any, second_branch: Any) -> Output:
        if first_branch is not None:
            return MergeLoop.Output(value=first_branch)  # type: ignore
        if second_branch is not None:
            return MergeLoop.Output(value=second_branch)  # type: ignore
        return MergeLoop.Output(value=None)  # type: ignore


def test_merge_default():
    component = MergeLoop(expected_type=int)
    results = component.run(first_branch=5, second_branch=None)
    assert results == component.Output(value=5)

    results = component.run(first_branch=None, second_branch=5)
    assert results == component.Output(value=5)

    results = component.run(first_branch=None, second_branch=None)
    assert results == component.Output(value=None)

    assert component.init_parameters == {}
