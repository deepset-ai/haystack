import json
import sys
import tempfile

import pytest

from haystack.components.converters import OpenAPIServiceToFunctions
from haystack.dataclasses import ByteStream


@pytest.fixture
def json_serperdev_openapi_spec():
    serper_spec = """
            {
                "openapi": "3.0.0",
                "info": {
                    "title": "SerperDev",
                    "version": "1.0.0",
                    "description": "API for performing search queries"
                },
                "servers": [
                    {
                        "url": "https://google.serper.dev"
                    }
                ],
                "paths": {
                    "/search": {
                        "post": {
                            "operationId": "search",
                            "description": "Search the web with Google",
                            "requestBody": {
                                "required": true,
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "q": {
                                                    "type": "string"
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "responses": {
                                "200": {
                                    "description": "Successful response",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "searchParameters": {
                                                        "type": "undefined"
                                                    },
                                                    "knowledgeGraph": {
                                                        "type": "undefined"
                                                    },
                                                    "answerBox": {
                                                        "type": "undefined"
                                                    },
                                                    "organic": {
                                                        "type": "undefined"
                                                    },
                                                    "topStories": {
                                                        "type": "undefined"
                                                    },
                                                    "peopleAlsoAsk": {
                                                        "type": "undefined"
                                                    },
                                                    "relatedSearches": {
                                                        "type": "undefined"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "security": [
                                {
                                    "apikey": []
                                }
                            ]
                        }
                    }
                },
                "components": {
                    "securitySchemes": {
                        "apikey": {
                            "type": "apiKey",
                            "name": "x-api-key",
                            "in": "header"
                        }
                    }
                }
            }
            """
    return serper_spec


@pytest.fixture
def yaml_serperdev_openapi_spec():
    serper_spec = """
            openapi: 3.0.0
            info:
              title: SerperDev
              version: 1.0.0
              description: API for performing search queries
            servers:
              - url: 'https://google.serper.dev'
            paths:
              /search:
                post:
                  operationId: search
                  description: Search the web with Google
                  requestBody:
                    required: true
                    content:
                      application/json:
                        schema:
                          type: object
                          properties:
                            q:
                              type: string
                  responses:
                    '200':
                      description: Successful response
                      content:
                        application/json:
                          schema:
                            type: object
                            properties:
                              searchParameters:
                                type: undefined
                              knowledgeGraph:
                                type: undefined
                              answerBox:
                                type: undefined
                              organic:
                                type: undefined
                              topStories:
                                type: undefined
                              peopleAlsoAsk:
                                type: undefined
                              relatedSearches:
                                type: undefined
                  security:
                    - apikey: []
            components:
              securitySchemes:
                apikey:
                  type: apiKey
                  name: x-api-key
                  in: header
            """
    return serper_spec


class TestOpenAPIServiceToFunctions:
    # test we can parse openapi spec given in json
    def test_openapi_spec_parsing_json(self, json_serperdev_openapi_spec):
        service = OpenAPIServiceToFunctions()

        serper_spec_json = service._parse_openapi_spec(json_serperdev_openapi_spec)
        assert serper_spec_json["openapi"] == "3.0.0"
        assert serper_spec_json["info"]["title"] == "SerperDev"

    # test we can parse openapi spec given in yaml
    def test_openapi_spec_parsing_yaml(self, yaml_serperdev_openapi_spec):
        service = OpenAPIServiceToFunctions()

        serper_spec_yaml = service._parse_openapi_spec(yaml_serperdev_openapi_spec)
        assert serper_spec_yaml["openapi"] == "3.0.0"
        assert serper_spec_yaml["info"]["title"] == "SerperDev"

    # test we can extract functions from openapi spec given
    def test_run_with_bytestream_source(self, json_serperdev_openapi_spec):
        service = OpenAPIServiceToFunctions()
        spec_stream = ByteStream.from_string(json_serperdev_openapi_spec)
        result = service.run(sources=[spec_stream], system_messages=["Some system message we don't care about here"])
        assert len(result["documents"]) == 1
        doc = result["documents"][0]

        # check that the content is as expected
        assert (
            doc.content
            == '{"name": "search", "description": "Search the web with Google", "parameters": {"type": "object", '
            '"properties": {"q": {"type": "string"}}}}'
        )

        # check that the metadata is as expected
        assert doc.meta["system_message"] == "Some system message we don't care about here"
        assert doc.meta["spec"] == json.loads(json_serperdev_openapi_spec)

    @pytest.mark.skipif(
        sys.platform in ["win32", "cygwin"],
        reason="Can't run on Windows Github CI, need access temp file but windows does not allow it",
    )
    def test_run_with_file_source(self, json_serperdev_openapi_spec):
        # test we can extract functions from openapi spec given in file
        service = OpenAPIServiceToFunctions()
        # write the spec to NamedTemporaryFile and check that it is parsed correctly
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(json_serperdev_openapi_spec.encode("utf-8"))
            tmp.seek(0)
            result = service.run(sources=[tmp.name], system_messages=["Some system message we don't care about here"])
            assert len(result["documents"]) == 1
            doc = result["documents"][0]

            # check that the content is as expected
            assert (
                doc.content
                == '{"name": "search", "description": "Search the web with Google", "parameters": {"type": "object", '
                '"properties": {"q": {"type": "string"}}}}'
            )

            # check that the metadata is as expected
            assert doc.meta["system_message"] == "Some system message we don't care about here"
            assert doc.meta["spec"] == json.loads(json_serperdev_openapi_spec)

    def test_run_with_file_source_and_none_system_messages(self, json_serperdev_openapi_spec):
        service = OpenAPIServiceToFunctions()
        spec_stream = ByteStream.from_string(json_serperdev_openapi_spec)

        # we now omit the system_messages argument
        result = service.run(sources=[spec_stream])
        assert len(result["documents"]) == 1
        doc = result["documents"][0]

        # check that the content is as expected
        assert (
            doc.content
            == '{"name": "search", "description": "Search the web with Google", "parameters": {"type": "object", '
            '"properties": {"q": {"type": "string"}}}}'
        )

        # check that the metadata is as expected, system_message should not be present
        assert "system_message" not in doc.meta
        assert doc.meta["spec"] == json.loads(json_serperdev_openapi_spec)

    def test_greet_oas_to_openai_function(self, test_files_path):
        service = OpenAPIServiceToFunctions()
        result = service.run(sources=[test_files_path / "yaml" / "openapi_greeting_service.yml"])
        assert len(result["documents"]) == 3
        mix_doc = result["documents"][0]

        correct_openai_fc_schema = {
            "name": "greet",
            "description": "Greet a person with a message (Mixed params and body)",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Custom message to send"},
                    "name": {"type": "string", "description": "Name of the person to greet"},
                },
                "required": ["name"],
            },
        }
        assert mix_doc.content == json.dumps(correct_openai_fc_schema)

        params_doc = result["documents"][1]
        correct_openai_fc_schema = {
            "name": "greetParams",
            "description": "Greet a person using URL parameter",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the person to greet using URL parameter"}
                },
                "required": ["name"],
            },
        }
        assert params_doc.content == json.dumps(correct_openai_fc_schema)

        body_doc = result["documents"][2]
        correct_openai_fc_schema = {
            "name": "greetBody",
            "description": "Greet a person with a message using JSON body only",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the person to greet"},
                    "message": {"type": "string", "description": "Custom message to send (optional)"},
                },
            },
        }
        assert body_doc.content == json.dumps(correct_openai_fc_schema)
