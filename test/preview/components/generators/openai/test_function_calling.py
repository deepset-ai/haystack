import json

import pytest

from haystack.preview.components.generators.openai.function_calling import FunctionCallingServiceContainer


@pytest.mark.unit
def test_generate_openai_function_descriptions():
    class WeatherService:
        def get_current_weather(self, city: str, unit: str = "celsius") -> str:
            """
            Get the current weather in a given location
            :param city: The city and state, e.g. San Francisco, CA
            :param unit: The unit to return the temperature in, celsius or fahrenheit
            :return: A structured JSON string containing the current weather information
            """
            weather_info = {"location": city, "temperature": "30", "unit": unit, "forecast": ["sunny", "windy"]}
            return json.dumps(weather_info)

    sc = FunctionCallingServiceContainer(WeatherService())
    assert sc.generate_openai_function_descriptions() == [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        }
    ]
