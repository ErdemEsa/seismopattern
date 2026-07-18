import pytest


@pytest.fixture(scope="session")
def client():
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_status_smoke(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.get_json()
    assert isinstance(data, dict)
    assert data.get("status") == "ok"
    assert "version" in data
    assert "models" in data


def test_docs_smoke(client):
    r = client.get("/docs")
    assert r.status_code == 200
    assert b"SwaggerUIBundle" in r.data


def test_openapi_smoke(client):
    r = client.get("/api/openapi.json")
    assert r.status_code == 200
    data = r.get_json()
    assert "paths" in data
    assert "/api/status" in data["paths"]
