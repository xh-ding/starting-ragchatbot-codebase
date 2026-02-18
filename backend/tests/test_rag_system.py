"""
Tests for RAGSystem.query() in rag_system.py.

These are integration-level tests (components mocked, not ChromaDB/Anthropic).
They verify the query orchestration: AI gets tools, session is updated, sources
are returned then reset, and — crucially — that config.MAX_RESULTS is nonzero
so ChromaDB actually returns results.
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_config(max_results=5):
    cfg = MagicMock()
    cfg.CHUNK_SIZE = 800
    cfg.CHUNK_OVERLAP = 100
    cfg.MAX_RESULTS = max_results
    cfg.MAX_HISTORY = 2
    cfg.CHROMA_PATH = "./test_chroma_db"
    cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    cfg.ANTHROPIC_API_KEY = "test-key"
    cfg.ANTHROPIC_MODEL = "claude-test"
    return cfg


def build_rag_system(config, ai_response="Here is the answer."):
    """
    Build a RAGSystem with all external dependencies mocked.
    Returns (system, mock_ai_generator).
    """
    with patch("rag_system.DocumentProcessor"), \
         patch("rag_system.VectorStore"), \
         patch("rag_system.AIGenerator") as mock_gen_cls, \
         patch("rag_system.SessionManager"):

        mock_gen = mock_gen_cls.return_value
        mock_gen.generate_response.return_value = ai_response

        from rag_system import RAGSystem
        system = RAGSystem(config)

    return system, mock_gen


# ---------------------------------------------------------------------------
# Tests: return value shape
# ---------------------------------------------------------------------------

class TestRAGSystemQueryReturnValue:

    def test_query_returns_two_element_tuple(self):
        """query() must return a (response_str, sources_list) tuple."""
        config = make_config()
        with patch("rag_system.DocumentProcessor"), \
             patch("rag_system.VectorStore"), \
             patch("rag_system.AIGenerator") as mock_gen_cls, \
             patch("rag_system.SessionManager"):

            mock_gen_cls.return_value.generate_response.return_value = "answer"

            from rag_system import RAGSystem
            system = RAGSystem(config)
            result = system.query("What is MCP?")

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_query_first_element_is_string(self):
        """First element of the tuple must be a string."""
        config = make_config()
        with patch("rag_system.DocumentProcessor"), \
             patch("rag_system.VectorStore"), \
             patch("rag_system.AIGenerator") as mock_gen_cls, \
             patch("rag_system.SessionManager"):

            mock_gen_cls.return_value.generate_response.return_value = "answer text"

            from rag_system import RAGSystem
            system = RAGSystem(config)
            response, _ = system.query("question")

        assert isinstance(response, str)
        assert response == "answer text"

    def test_query_second_element_is_list(self):
        """Second element (sources) must be a list."""
        config = make_config()
        with patch("rag_system.DocumentProcessor"), \
             patch("rag_system.VectorStore"), \
             patch("rag_system.AIGenerator") as mock_gen_cls, \
             patch("rag_system.SessionManager"):

            mock_gen_cls.return_value.generate_response.return_value = "ok"

            from rag_system import RAGSystem
            system = RAGSystem(config)
            _, sources = system.query("question")

        assert isinstance(sources, list)


# ---------------------------------------------------------------------------
# Tests: tools are wired up
# ---------------------------------------------------------------------------

class TestRAGSystemToolUsage:

    def test_query_passes_tool_definitions_to_ai_generator(self):
        """generate_response() must receive non-empty 'tools' for content questions."""
        config = make_config()
        with patch("rag_system.DocumentProcessor"), \
             patch("rag_system.VectorStore"), \
             patch("rag_system.AIGenerator") as mock_gen_cls, \
             patch("rag_system.SessionManager"):

            mock_gen = mock_gen_cls.return_value
            mock_gen.generate_response.return_value = "answer"

            from rag_system import RAGSystem
            system = RAGSystem(config)
            system.query("What does lesson 3 cover?")

        call_kwargs = mock_gen.generate_response.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is not None
        assert len(call_kwargs["tools"]) > 0

    def test_query_passes_tool_manager_to_ai_generator(self):
        """generate_response() must receive a tool_manager so it can execute tools."""
        config = make_config()
        with patch("rag_system.DocumentProcessor"), \
             patch("rag_system.VectorStore"), \
             patch("rag_system.AIGenerator") as mock_gen_cls, \
             patch("rag_system.SessionManager"):

            mock_gen = mock_gen_cls.return_value
            mock_gen.generate_response.return_value = "answer"

            from rag_system import RAGSystem
            system = RAGSystem(config)
            system.query("course question")

        call_kwargs = mock_gen.generate_response.call_args[1]
        assert "tool_manager" in call_kwargs
        assert call_kwargs["tool_manager"] is not None


# ---------------------------------------------------------------------------
# Tests: session management
# ---------------------------------------------------------------------------

class TestRAGSystemSessionManagement:

    def test_query_updates_session_history_when_session_provided(self):
        """query() must call add_exchange() with the user's question and AI response."""
        config = make_config()
        with patch("rag_system.DocumentProcessor"), \
             patch("rag_system.VectorStore"), \
             patch("rag_system.AIGenerator") as mock_gen_cls, \
             patch("rag_system.SessionManager") as mock_sess_cls:

            mock_gen_cls.return_value.generate_response.return_value = "AI answer"
            mock_sess = mock_sess_cls.return_value

            from rag_system import RAGSystem
            system = RAGSystem(config)
            system.query("user question", session_id="session_1")

        mock_sess.add_exchange.assert_called_once_with(
            "session_1", "user question", "AI answer"
        )

    def test_query_does_not_update_history_without_session(self):
        """When no session_id is given, add_exchange() must not be called."""
        config = make_config()
        with patch("rag_system.DocumentProcessor"), \
             patch("rag_system.VectorStore"), \
             patch("rag_system.AIGenerator") as mock_gen_cls, \
             patch("rag_system.SessionManager") as mock_sess_cls:

            mock_gen_cls.return_value.generate_response.return_value = "answer"
            mock_sess = mock_sess_cls.return_value

            from rag_system import RAGSystem
            system = RAGSystem(config)
            system.query("stateless question")

        mock_sess.add_exchange.assert_not_called()

    def test_query_fetches_conversation_history_for_session(self):
        """query() must retrieve conversation history and pass it to generate_response()."""
        config = make_config()
        with patch("rag_system.DocumentProcessor"), \
             patch("rag_system.VectorStore"), \
             patch("rag_system.AIGenerator") as mock_gen_cls, \
             patch("rag_system.SessionManager") as mock_sess_cls:

            mock_gen = mock_gen_cls.return_value
            mock_gen.generate_response.return_value = "answer"
            mock_sess = mock_sess_cls.return_value
            mock_sess.get_conversation_history.return_value = "User: prev\nAssistant: ok"

            from rag_system import RAGSystem
            system = RAGSystem(config)
            system.query("follow-up", session_id="s1")

        mock_sess.get_conversation_history.assert_called_once_with("s1")
        gen_kwargs = mock_gen.generate_response.call_args[1]
        assert gen_kwargs["conversation_history"] == "User: prev\nAssistant: ok"


# ---------------------------------------------------------------------------
# Tests: source reset
# ---------------------------------------------------------------------------

class TestRAGSystemSources:

    def test_sources_reset_after_each_query(self):
        """
        After query() completes, search_tool.last_sources must be empty.
        This prevents sources from leaking from one response to the next.
        """
        config = make_config()
        with patch("rag_system.DocumentProcessor"), \
             patch("rag_system.VectorStore"), \
             patch("rag_system.AIGenerator") as mock_gen_cls, \
             patch("rag_system.SessionManager"):

            mock_gen_cls.return_value.generate_response.return_value = "answer"

            from rag_system import RAGSystem
            system = RAGSystem(config)
            # Simulate a tool search having run
            system.search_tool.last_sources = [{"label": "stale", "url": None}]

            system.query("any question")

        assert system.search_tool.last_sources == [], (
            "last_sources must be reset after query() so old sources don't "
            "appear on future responses."
        )

    def test_sources_returned_from_last_search_tool_call(self):
        """Sources collected during the query must be returned to the caller."""
        config = make_config()
        expected_sources = [{"label": "Course A - Lesson 1", "url": "https://example.com"}]

        with patch("rag_system.DocumentProcessor"), \
             patch("rag_system.VectorStore"), \
             patch("rag_system.AIGenerator") as mock_gen_cls, \
             patch("rag_system.SessionManager"):

            mock_gen_cls.return_value.generate_response.return_value = "answer"

            from rag_system import RAGSystem
            system = RAGSystem(config)
            # Simulate the search tool having populated sources during generate_response
            system.search_tool.last_sources = expected_sources

            _, sources = system.query("course question")

        assert sources == expected_sources


# ---------------------------------------------------------------------------
# Tests: configuration — the primary root-cause check
# ---------------------------------------------------------------------------

class TestRAGSystemConfiguration:

    def test_config_max_results_is_nonzero(self):
        """
        REGRESSION TEST: config.MAX_RESULTS must be > 0.

        With MAX_RESULTS = 0, ChromaDB's n_results argument is 0, so it always
        returns an empty document list. CourseSearchTool.execute() then returns
        'No relevant content found' for every query, which manifests as the
        'query failed' symptom reported by the user.

        Expected value per CLAUDE.md: 5.
        """
        from config import config  # the real singleton

        assert config.MAX_RESULTS > 0, (
            f"config.MAX_RESULTS = {config.MAX_RESULTS}. "
            "It must be > 0 (ideally 5) for ChromaDB to return search results. "
            "All content questions will silently fail when this is 0."
        )

    def test_config_max_results_is_at_least_five(self):
        """MAX_RESULTS should be at least 5 to return meaningful search results."""
        from config import config

        assert config.MAX_RESULTS >= 5, (
            f"config.MAX_RESULTS = {config.MAX_RESULTS}. "
            "CLAUDE.md specifies the intended default is 5."
        )
