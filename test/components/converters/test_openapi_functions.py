from haystack.components.converters import OpenAPIServiceToFunctions


class TestOpenAPIServiceToFunctions:
    #  Extract functions from OpenAPI specification and convert them into a format suitable for OpenAI function calling
    def test_extract_functions_from_openapi_specification(self):
        service = OpenAPIServiceToFunctions()

        # Define a sample OpenAPI specification
        openapi_spec = {
            "paths": {
                "/path1": {
                    "get": {
                        "operationId": "function1",
                        "description": "Function 1 description",
                        "parameters": [{"name": "param1", "schema": {"type": "string"}}],
                    }
                },
                "/path2": {
                    "post": {
                        "operationId": "function2",
                        "summary": "Function 2 summary",
                        "parameters": [{"name": "param2", "schema": {"type": "integer"}}],
                    }
                },
            }
        }

        functions = service._openapi_to_functions(openapi_spec)

        # Assert that the functions list contains the expected function definitions
        assert functions == [
            {
                "name": "function1",
                "description": "Function 1 description",
                "parameters": {
                    "type": "object",
                    "properties": {"parameters": {"type": "object", "properties": {"param1": {"type": "string"}}}},
                },
            },
            {
                "name": "function2",
                "description": "Function 2 summary",
                "parameters": {
                    "type": "object",
                    "properties": {"parameters": {"type": "object", "properties": {"param2": {"type": "integer"}}}},
                },
            },
        ]
