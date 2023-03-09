from typing import Dict, Any, List, Tuple
import logging

from canals import node


logger = logging.getLogger(__name__)


@node
class Greet:
    def __init__(self, message: str = "\nGreeting node says: Hi!\n", edge: str = "value"):
        """
        Logs a greeting message without affecting the value
        passing on the edge.

        Single input, single output node.

        :param message: the message to print. This is also a parameter.
        :param edge: the name of the input and output edge.
        """
        self.message = message

        self.init_parameters = {"edge": edge, "message": message}
        self.inputs = [edge]
        self.outputs = [edge]

    def run(self, name: str, data: List[Tuple[str, Any]], parameters: Dict[str, Any]):
        message = parameters.get(name, {}).get("message", self.message)
        logger.info(message)
        return ({data[0][0]: data[0][1]}, parameters)


def test_greet_default(caplog):
    caplog.set_level(logging.INFO)
    node = Greet()
    results = node.run(name="test_node", data=[("value", 10)], parameters={})
    assert results == ({"value": 10}, {})
    assert "Greeting node says: Hi!" in caplog.text
    assert node.init_parameters == {"edge": "value", "message": "\nGreeting node says: Hi!\n"}


def test_greet_init_params(caplog):
    caplog.set_level(logging.INFO)
    node = Greet(edge="test", message="Hello!")
    results = node.run(name="test_node", data=[("test", 10)], parameters={})
    assert results == ({"test": 10}, {})
    assert "Hello" in caplog.text
    assert node.init_parameters == {"edge": "test", "message": "Hello!"}


def test_greet_runtime_params(caplog):
    caplog.set_level(logging.INFO)
    node = Greet(edge="test", message="Hello!")
    results = node.run(name="test_node", data=[("test", 10)], parameters={"test_node": {"message": "Ciao :)"}})
    assert results == ({"test": 10}, {"test_node": {"message": "Ciao :)"}})
    assert "Ciao :)" in caplog.text
    assert node.init_parameters == {"edge": "test", "message": "Hello!"}
