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
            '"properties": {"requestBody": {"type": "object", "properties": {"q": {"type": "string"}}}}}}'
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
                '"properties": {"requestBody": {"type": "object", "properties": {"q": {"type": "string"}}}}}}'
            )

            # check that the metadata is as expected
            assert doc.meta["system_message"] == "Some system message we don't care about here"
            assert doc.meta["spec"] == json.loads(json_serperdev_openapi_spec)
