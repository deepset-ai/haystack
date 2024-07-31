from typing import Any, Callable, Dict


class Tool:
    def __init__(self, name: str, description: str, parameters: Dict[str, Any], function: Callable):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function

    @property
    def tool_spec(self):
        """Get the tool specification."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def invoke(self, **kwargs):
        """Invoke the tool."""
        return self.function(**kwargs)
