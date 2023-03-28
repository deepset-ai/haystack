import pytest

from haystack.document_stores.memory_knowledgegraph import InMemoryKnowledgeGraph

from ..conftest import fail_at_version


@pytest.mark.unit
@fail_at_version(1, 17)
def test_in_memory_knowledge_graph_deprecation_warning():
    with pytest.warns(DeprecationWarning) as w:
        InMemoryKnowledgeGraph()

        assert len(w) == 2
        assert (
            w[0].message.args[0]
            == "The InMemoryKnowledgeGraph component is deprecated and will be removed in future versions."
        )
        assert (
            w[1].message.args[0]
            == "The BaseKnowledgeGraph component is deprecated and will be removed in future versions."
        )
