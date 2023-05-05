from typing import Any

from dataclasses import make_dataclass

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

        # Here is where the run method is defined with the specific types.
        # Let's minimize the amount of logic contained here to the absolute minimum, because this area of the code
        # is really complex to reason about.

        def run(self, *value: expected_type):
            return self._run(*value)

        cls.run = run

        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        obj.expected_type = expected_type
        return obj


@component
class MergeLoop(metaclass=_MetaClass):
    """
    Takes two input components and returns the first one that is not None.

    In case both received a value, priority is given to 'first'.

    Always initialize this class by passing the expected type, like: `MergeLoop(expected_type=int)`.
    """

    def run(self, *value: Any):
        """
        Takes some inputs and returns the first one that is not None.

        In case it receives more than one value, priority is given to the first.
        """
        #
        # Note: highly dynamically typed classes like MergeLoop are mostly built by the metaclass.
        # However, @component's reflexive checks will fail if the class does not have a 'run()' method,
        # so for now we have to provide these mocks to make it happy.
        #
        # The real run() is defined in '_MetaClass.__call__()'. Note how 'run()' in fact calls back immediately
        # '_run()' to avoid encoding too much logic in the metaclass itself.
        #
        # TODO generalize this metaclass to spare contributors from understanding this thing.
        #
        pass

    @property
    def output_type(self):
        return make_dataclass("Output", [(f"value", self.expected_type, None)])

    def _run(self, *value: Any):
        for v in value:
            if v is not None:
                return self.output_type(value=v)
        return self.output_type(value=None)


def test_merge():
    component = MergeLoop(expected_type=int)
    results = component.run(5, None)
    assert results.__dict__ == component.output_type(value=5).__dict__

    results = component.run(None, 5)
    assert results.__dict__ == component.output_type(value=5).__dict__

    results = component.run(None, None, None)
    assert results.__dict__ == component.output_type(value=None).__dict__

    results = component.run()
    assert results.__dict__ == component.output_type(value=None).__dict__
