import pytest
from fastapi.testclient import TestClient
import os
import sqlite3

# Clean up database before tests
@pytest.fixture(scope="session", autouse=True)
def setup_db():
    if os.path.exists('urls.db'):
        os.remove('urls.db')
    yield
    if os.path.exists('urls.db'):
        os.remove('urls.db')

@pytest.fixture
def client():
    from main import app
    return TestClient(app)


def test_index_page(client):
    """Test the home page loads"""
    response = client.get("/")
    assert response.status_code == 200
    assert "URL Shortener" in response.text


def test_health_check(client):
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_shorten_valid_url(client):
    """Test shortening a valid URL"""
    response = client.post("/shorten", json={
        "url": "https://www.example.com/very/long/url/path"
    })
    assert response.status_code == 200
    data = response.json()
    assert "short_code" in data
    assert len(data["short_code"]) == 6
    assert data["original_url"] == "https://www.example.com/very/long/url/path"


def test_shorten_invalid_url(client):
    """Test shortening an invalid URL"""
    response = client.post("/shorten", json={
        "url": "not-a-valid-url"
    })
    assert response.status_code == 400
    assert "Invalid URL" in response.json()["detail"]


def test_shorten_missing_url(client):
    """Test shortening with missing URL"""
    response = client.post("/shorten", json={})
    assert response.status_code == 400
    assert "required" in response.json()["detail"]


def test_redirect(client):
    """Test redirect endpoint"""
    # First create a short URL
    create_response = client.post("/shorten", json={
        "url": "https://www.example.com/test"
    })
    short_code = create_response.json()["short_code"]
    
    # Now test redirect
    response = client.get(f"/{short_code}", follow_redirects=False)
    assert response.status_code == 307
    assert "https://www.example.com/test" in response.headers["location"]


def test_redirect_not_found(client):
    """Test redirect with invalid code"""
    response = client.get("/invalidcode", follow_redirects=False)
    assert response.status_code == 404


def test_stats(client):
    """Test stats endpoint"""
    # Create a short URL
    create_response = client.post("/shorten", json={
        "url": "https://www.example.com/stats-test"
    })
    short_code = create_response.json()["short_code"]
    
    # Access stats
    response = client.get(f"/stats/{short_code}")
    assert response.status_code == 200
    data = response.json()
    assert data["short_code"] == short_code
    assert data["original_url"] == "https://www.example.com/stats-test"
    assert data["click_count"] == 0
    assert "created_at" in data


def test_stats_not_found(client):
    """Test stats with invalid code"""
    response = client.get("/stats/invalidcode")
    assert response.status_code == 404


def test_click_count_increment(client):
    """Test that click count increases after redirect"""
    # Create a short URL
    create_response = client.post("/shorten", json={
        "url": "https://www.example.com/click-test"
    })
    short_code = create_response.json()["short_code"]
    
    # Get initial stats
    stats_response = client.get(f"/stats/{short_code}")
    initial_count = stats_response.json()["click_count"]
    
    # Access the short URL (redirect)
    client.get(f"/{short_code}", follow_redirects=False)
    
    # Check stats again
    stats_response = client.get(f"/stats/{short_code}")
    new_count = stats_response.json()["click_count"]
    
    assert new_count == initial_count + 1


def test_url_validation_http(client):
    """Test that HTTP URLs are accepted"""
    response = client.post("/shorten", json={
        "url": "http://example.com/test"
    })
    assert response.status_code == 200


def test_url_validation_localhost(client):
    """Test that localhost URLs are accepted"""
    response = client.post("/shorten", json={
        "url": "http://localhost:8000/test"
    })
    assert response.status_code == 200
