# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack.components.builders.hierarchical_doc_builder import HierarchicalDocumentBuilder


class TestHierarchicalDocumentBuilder:
    def test_init_with_default_params(self):
        builder = HierarchicalDocumentBuilder(block_sizes=[100, 200, 300])
        assert builder.block_sizes == [300, 200, 100]
        assert builder.split_overlap == 0
        assert builder.split_by == "word"

    def test_init_with_custom_params(self):
        builder = HierarchicalDocumentBuilder(block_sizes=[100, 200, 300], split_overlap=25, split_by="word")
        assert builder.block_sizes == [300, 200, 100]
        assert builder.split_overlap == 25
        assert builder.split_by == "word"

    def test_init_with_duplicate_block_sizes(self):
        try:
            HierarchicalDocumentBuilder(block_sizes=[100, 200, 200])
        except ValueError as e:
            assert str(e) == "block_sizes must not contain duplicates"

    def test_to_dict(self):
        builder = HierarchicalDocumentBuilder(block_sizes=[100, 200, 300], split_overlap=25, split_by="word")
        expected = builder.to_dict()
        assert expected == {
            "type": "haystack.components.builders.hierarchical_doc_builder.HierarchicalDocumentBuilder",
            "init_parameters": {"block_sizes": [300, 200, 100], "split_overlap": 25, "split_by": "word"},
        }

    """
    def test_to_dict_without_optional_params(self):
        builder = PromptBuilder(template="This is a {{ variable }}")
        res = builder.to_dict()
        assert res == {
            "type": "haystack.components.builders.prompt_builder.PromptBuilder",
            "init_parameters": {"template": "This is a {{ variable }}", "variables": None, "required_variables": None},
        }

    def test_run(self):
        builder = PromptBuilder(template="This is a {{ variable }}")
        res = builder.run(variable="test")
        assert res == {"prompt": "This is a test"}

    def test_example_in_pipeline(self):
        default_template = "Here is the document: {{documents[0].content}} \\n Answer: {{query}}"
        prompt_builder = PromptBuilder(template=default_template, variables=["documents"])

        @component
        class DocumentProducer:
            @component.output_types(documents=List[Document])
            def run(self, doc_input: str):
                return {"documents": [Document(content=doc_input)]}

        pipe = Pipeline()
        pipe.add_component("doc_producer", DocumentProducer())
        pipe.add_component("prompt_builder", prompt_builder)
        pipe.connect("doc_producer.documents", "prompt_builder.documents")

        template = "Here is the document: {{documents[0].content}} \n Query: {{query}}"
        result = pipe.run(
            data={
                "doc_producer": {"doc_input": "Hello world, I live in Berlin"},
                "prompt_builder": {
                    "template": template,
                    "template_variables": {"query": "Where does the speaker live?"},
                },
            }
        )

        assert result == {
            "prompt_builder": {
                "prompt": "Here is the document: Hello world, I live in Berlin \n Query: Where does the speaker live?"
            }
        }

    def test_example_in_pipeline_simple(self):
        default_template = "This is the default prompt:\n Query: {{query}}"
        prompt_builder = PromptBuilder(template=default_template)

        pipe = Pipeline()
        pipe.add_component("prompt_builder", prompt_builder)

        # using the default prompt
        result = pipe.run(data={"query": "Where does the speaker live?"})
        expected_default = {
            "prompt_builder": {"prompt": "This is the default prompt:\n Query: Where does the speaker live?"}
        }
        assert result == expected_default

        # using the dynamic prompt
        result = pipe.run(
            data={"query": "Where does the speaker live?", "template": "This is the dynamic prompt:\n Query: {{query}}"}
        )
        expected_dynamic = {
            "prompt_builder": {"prompt": "This is the dynamic prompt:\n Query: Where does the speaker live?"}
        }
        assert result == expected_dynamic
    """
