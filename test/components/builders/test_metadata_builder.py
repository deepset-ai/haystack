import pytest
from haystack.dataclasses import Document
from haystack.components.builders import MetadataBuilder


class TestMetadataBuilder:
    def test_recieves_list_of_summaries_entities(self):
        metadata_builder = MetadataBuilder(meta_keys=["entities", "summary"])
        documents = [Document(content="document_0"), Document(content="document_1")]
        data = {"entities": ["entity1", "entity2", "entity3"], "summary": ["Summary 1", "Summary 2", "Summary3"]}
        metadata = [{"key_0": "value_0"}, {"key_1": "value_1"}]

        result = metadata_builder.run(documents=documents, data=data, meta=metadata)
        assert len(result["documents"]) == 2

    def test_receives_list_of_replies_and_no_metadata(self):
        """
        The component receives only a list of Documents and replies and no metadata.
        """
        metadata_builder = MetadataBuilder(meta_keys=["summary"])

        documents = [Document(content="document_0")]
        data = {"summary": ["reply_0", "reply_1", "reply_2"]}
        meta = [{"key_0": "value_0"}]
        # Invoke the run method without providing metadata
        result = metadata_builder.run(documents=documents, data=data, meta=meta)

        assert isinstance(result["documents"], list)
        assert len(result["documents"]) == 1

    def test_receives_list_of_replies_and_metadata(self):
        """
        The component receives a list of Documents, replies and metadata.
        """
        metadata_builder = MetadataBuilder(meta_keys=["replies"])
        # Single document, one
        documents = [Document(content="document_0")]
        data = {"replies": ["reply1", "reply2"]}
        metadata = [{"key_0": "value_0"}]

        result = metadata_builder.run(documents=documents, meta=metadata, data=data)

        assert isinstance(result["documents"], list)
        assert len(result["documents"]) == 1

    def test_recieves_replies_and_no_metadata(self):
        """
        The component receives only a list of Documents and replies and no metadata.
        """
        metadata_builder = MetadataBuilder(meta_keys=["replies"])

        documents = [Document(content="document_0")]
        data = {"replies": ["reply1", "reply2"]}

        # Invoke the run method without providing metadata
        result = metadata_builder.run(documents=documents, data=data)

        assert isinstance(result["documents"], list)
        assert len(result["documents"]) == 1

    def test_mismatched_documents_replies_and_no_metadata(self):
        """
        If the length of the Document list and the replies list are different having no metadata, the component raises a ValueError.
        """
        metadata_builder = MetadataBuilder(meta_keys=["replies"])
        documents = [Document(content="document_0"), Document(content="document_1")]
        data = {"replies": ["reply1", "reply2", "reply3"]}

        # Check that a ValueError is raised when invoking the run method
        with pytest.raises(ValueError):
            metadata_builder.run(documents=documents, data=data)

    def test_mismatched_documents_replies(self):
        """
        If the length of the Document list and the replies list are different, having metadata the component raises a ValueError.
        """
        metadata_builder = MetadataBuilder(meta_keys=["replies"])

        documents = [Document(content="document_0"), Document(content="document_1")]
        data = {"replies": ["reply1", "reply2", "reply3"]}
        metadata = [{"key_0": "value_0"}, {"key_1": "value_1"}, {"key_2": "value_2"}]

        # Check that a ValueError is raised when invoking the run method
        with pytest.raises(ValueError):
            metadata_builder.run(documents=documents, data=data, meta=metadata)

    def test_mismatched_documents_metadata(self):
        """
        If the length of the Document list and the metadata list are different, the component raises a ValueError.
        """
        metadata_builder = MetadataBuilder(meta_keys=["replies"])

        documents = [Document(content="document_0"), Document(content="document_1"), Document(content="document_2")]
        data = {"replies": ["reply1", "reply2", "reply3"]}
        metadata = [{"key_0": "value_0"}, {"key_1": "value_1"}]

        # Check that a ValueError is raised when invoking the run method
        with pytest.raises(ValueError):
            metadata_builder.run(documents=documents, data=data, meta=metadata)

    def test_mismatched_documents_replies_metadata(self):
        """
        If the length of the Document list, replies list and the metadata list are all different, the component raises a ValueError.
        """
        metadata_builder = MetadataBuilder(meta_keys=["replies"])

        documents = [Document(content="document_0"), Document(content="document_1")]
        data = {"replies": ["reply0"]}
        metadata = [{"key_0": "value_0"}, {"key_1": "value_1"}, {"key_2": "value_2"}]

        # Check that a ValueError is raised when invoking the run method
        with pytest.raises(ValueError):
            metadata_builder.run(documents=documents, data=data, meta=metadata)

    def test_metadata_with_same_keys(self):
        """
        The component should correctly add the metadata if the Document metadata already has a reply.
        """
        metadata_builder = MetadataBuilder(meta_keys=["replies"])
        data = {"replies": ["reply_0"]}
        documents = [Document(content="document content", meta={"reply": "original text"})]
        result = metadata_builder.run(documents=documents, data=data)

        assert isinstance(result["documents"], list)
        assert len(result["documents"]) == 1
