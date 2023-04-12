from typing import Dict, Any, List, Tuple
import logging

from canals import component


logger = logging.getLogger(__name__)


@component
class Greet:
    def __init__(self, message: str = "\nGreeting component says: Hi!\n", connection: str = "value"):
        """
        Logs a greeting message without affecting the value
        passing on the connection.

        Single input, single output component.

        :param message: the message to print. This is also a parameter.
        :param connection: the name of the input and output connection.
        """
        self.message = message

        self.init_parameters = {"connection": connection, "message": message}
        self.inputs = [connection]
        self.outputs = [connection]

    def run(self, name: str, data: List[Tuple[str, Any]], parameters: Dict[str, Any]):
        message = parameters.get(name, {}).get("message", self.message)
        logger.info(message)
        return ({data[0][0]: data[0][1]}, parameters)


def test_greet_default(caplog):
    caplog.set_level(logging.INFO)
    component = Greet()
    results = component.run(name="test_component", data=[("value", 10)], parameters={})
    assert results == ({"value": 10}, {})
    assert "Greeting component says: Hi!" in caplog.text
    assert component.init_parameters == {"connection": "value", "message": "\nGreeting component says: Hi!\n"}


def test_greet_init_params(caplog):
    caplog.set_level(logging.INFO)
    component = Greet(connection="test", message="Hello!")
    results = component.run(name="test_component", data=[("test", 10)], parameters={})
    assert results == ({"test": 10}, {})
    assert "Hello" in caplog.text
    assert component.init_parameters == {"connection": "test", "message": "Hello!"}


def test_greet_runtime_params(caplog):
    caplog.set_level(logging.INFO)
    component = Greet(connection="test", message="Hello!")
    results = component.run(
        name="test_component", data=[("test", 10)], parameters={"test_component": {"message": "Ciao :)"}}
    )
    assert results == ({"test": 10}, {"test_component": {"message": "Ciao :)"}})
    assert "Ciao :)" in caplog.text
    assert component.init_parameters == {"connection": "test", "message": "Hello!"}
