from haystack import __version__


def test_ready(client):
    response = client.get(url="/ready")
    assert 200 == response.status_code
    response_json = response.json()
    assert response_json == True


def test_version(client):
    response = client.get(url="/version")
    assert 200 == response.status_code
    response_json = response.json()
    assert response_json == {"haystack": __version__}
