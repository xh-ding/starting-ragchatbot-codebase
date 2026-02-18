"""
Tests for the FastAPI API endpoints.

Uses an inline test app that mirrors app.py's routes and injects a mock
RAGSystem. The static file mount from the real app is intentionally omitted
because the ../frontend directory does not exist in the test environment.
"""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import Any, List, Optional
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Inline test app — mirrors app.py routes without the static mount
# ---------------------------------------------------------------------------

def make_test_app(rag_system):
    """Return a FastAPI app wired to the given (mock) rag_system."""
    app = FastAPI()

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Any]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        try:
            rag_system.session_manager.clear_session(session_id)
            return {"status": "ok"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ---------------------------------------------------------------------------
# Shared client fixture (local to this module — uses conftest mock_rag_system)
# ---------------------------------------------------------------------------

@pytest.fixture
def client(mock_rag_system):
    app = make_test_app(mock_rag_system)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests: POST /api/query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:

    def test_returns_200_with_valid_query(self, client):
        """A well-formed POST /api/query returns HTTP 200."""
        resp = client.post("/api/query", json={"query": "What is MCP?"})
        assert resp.status_code == 200

    def test_response_contains_answer_field(self, client, mock_rag_system):
        """Response JSON must include an 'answer' string."""
        mock_rag_system.query.return_value = ("MCP is a protocol.", [])
        resp = client.post("/api/query", json={"query": "What is MCP?"})
        body = resp.json()
        assert "answer" in body
        assert body["answer"] == "MCP is a protocol."

    def test_response_contains_sources_list(self, client, mock_rag_system):
        """Response JSON must include a 'sources' list."""
        sources = [{"label": "Course A - Lesson 1", "url": "https://example.com"}]
        mock_rag_system.query.return_value = ("answer", sources)
        resp = client.post("/api/query", json={"query": "question"})
        body = resp.json()
        assert "sources" in body
        assert body["sources"] == sources

    def test_response_contains_session_id(self, client):
        """Response JSON must include a 'session_id' string."""
        resp = client.post("/api/query", json={"query": "test"})
        body = resp.json()
        assert "session_id" in body
        assert isinstance(body["session_id"], str)

    def test_auto_creates_session_when_none_provided(self, client, mock_rag_system):
        """When session_id is omitted, a new session is created via session_manager."""
        mock_rag_system.session_manager.create_session.return_value = "auto-session-99"
        resp = client.post("/api/query", json={"query": "new question"})
        mock_rag_system.session_manager.create_session.assert_called_once()
        assert resp.json()["session_id"] == "auto-session-99"

    def test_uses_provided_session_id(self, client, mock_rag_system):
        """When session_id is supplied, create_session() must NOT be called."""
        resp = client.post(
            "/api/query",
            json={"query": "follow-up", "session_id": "existing-session"},
        )
        mock_rag_system.session_manager.create_session.assert_not_called()
        assert resp.json()["session_id"] == "existing-session"

    def test_passes_query_and_session_id_to_rag_system(self, client, mock_rag_system):
        """rag_system.query() is called with the right query and session_id."""
        client.post(
            "/api/query",
            json={"query": "What is an agent?", "session_id": "s1"},
        )
        mock_rag_system.query.assert_called_once_with("What is an agent?", "s1")

    def test_returns_500_when_rag_system_raises(self, client, mock_rag_system):
        """If rag_system.query() raises, the endpoint returns HTTP 500."""
        mock_rag_system.query.side_effect = RuntimeError("DB unavailable")
        resp = client.post("/api/query", json={"query": "anything"})
        assert resp.status_code == 500

    def test_error_detail_included_in_500_response(self, client, mock_rag_system):
        """The 500 response body should include the exception message."""
        mock_rag_system.query.side_effect = RuntimeError("DB unavailable")
        resp = client.post("/api/query", json={"query": "anything"})
        assert "DB unavailable" in resp.json()["detail"]

    def test_missing_query_field_returns_422(self, client):
        """Omitting the required 'query' field returns HTTP 422 Unprocessable Entity."""
        resp = client.post("/api/query", json={"session_id": "s1"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Tests: GET /api/courses
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:

    def test_returns_200(self, client):
        """GET /api/courses returns HTTP 200."""
        resp = client.get("/api/courses")
        assert resp.status_code == 200

    def test_response_contains_total_courses(self, client, mock_rag_system):
        """Response JSON must include 'total_courses' as an integer."""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["A", "B", "C"],
        }
        body = client.get("/api/courses").json()
        assert "total_courses" in body
        assert body["total_courses"] == 3

    def test_response_contains_course_titles_list(self, client, mock_rag_system):
        """Response JSON must include 'course_titles' as a list of strings."""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Intro to ML", "Advanced NLP"],
        }
        body = client.get("/api/courses").json()
        assert "course_titles" in body
        assert body["course_titles"] == ["Intro to ML", "Advanced NLP"]

    def test_returns_500_when_analytics_raises(self, client, mock_rag_system):
        """If get_course_analytics() raises, the endpoint returns HTTP 500."""
        mock_rag_system.get_course_analytics.side_effect = Exception("analytics error")
        resp = client.get("/api/courses")
        assert resp.status_code == 500

    def test_zero_courses_is_valid_response(self, client, mock_rag_system):
        """An empty course catalog returns total_courses=0 and an empty list."""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        body = client.get("/api/courses").json()
        assert body["total_courses"] == 0
        assert body["course_titles"] == []


# ---------------------------------------------------------------------------
# Tests: DELETE /api/sessions/{session_id}
# ---------------------------------------------------------------------------

class TestDeleteSessionEndpoint:

    def test_returns_200_on_success(self, client):
        """DELETE /api/sessions/{id} returns HTTP 200."""
        resp = client.delete("/api/sessions/abc-123")
        assert resp.status_code == 200

    def test_response_body_is_status_ok(self, client):
        """Successful deletion returns {"status": "ok"}."""
        resp = client.delete("/api/sessions/abc-123")
        assert resp.json() == {"status": "ok"}

    def test_calls_clear_session_with_correct_id(self, client, mock_rag_system):
        """clear_session() is called with the session_id from the URL path."""
        client.delete("/api/sessions/my-session-42")
        mock_rag_system.session_manager.clear_session.assert_called_once_with(
            "my-session-42"
        )

    def test_returns_500_when_clear_session_raises(self, client, mock_rag_system):
        """If clear_session() raises, the endpoint returns HTTP 500."""
        mock_rag_system.session_manager.clear_session.side_effect = RuntimeError("fail")
        resp = client.delete("/api/sessions/bad-session")
        assert resp.status_code == 500
