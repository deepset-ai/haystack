def pytest_addoption(parser):
    parser.addoption(
        "--document_store_type", action="store", default="elasticsearch, faiss, sql, memory, milvus, weaviate, pinecone"
    )
    parser.addoption(
        "--mock-dc", action="store_true", default=True, help="Mock HTTP requests to dC while running tests"
    )
    parser.addoption(
        "--mock-pinecone", action="store_true", default=True, help="Mock HTTP requests to Pinecone while running tests"
    )


def pytest_generate_tests(metafunc):
    # Get selected docstores from CLI arg
    document_store_type = metafunc.config.option.document_store_type
    selected_doc_stores = [item.strip() for item in document_store_type.split(",")]

    # parametrize document_store fixture if it's in the test function argument list
    # but does not have an explicit parametrize annotation e.g
    # @pytest.mark.parametrize("document_store", ["memory"], indirect=False)
    found_mark_parametrize_document_store = False
    for marker in metafunc.definition.iter_markers("parametrize"):
        if "document_store" in marker.args[0]:
            found_mark_parametrize_document_store = True
            break
    # for all others that don't have explicit parametrization, we add the ones from the CLI arg
    if "document_store" in metafunc.fixturenames and not found_mark_parametrize_document_store:
        metafunc.parametrize("document_store", selected_doc_stores, indirect=True)
