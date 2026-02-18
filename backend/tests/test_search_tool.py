"""
Tests for CourseSearchTool.execute() in search_tools.py.

These tests mock VectorStore to control what search() returns, isolating
the search tool from ChromaDB. The tests expose the symptom caused by
config.MAX_RESULTS = 0 (which makes ChromaDB return zero results).
"""

import pytest
from unittest.mock import MagicMock
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


def make_mock_store(documents=None, metadata=None, distances=None, error=None):
    """Build a VectorStore mock that returns a controlled SearchResults."""
    store = MagicMock()
    result = SearchResults(
        documents=documents or [],
        metadata=metadata or [],
        distances=distances or [],
        error=error,
    )
    store.search.return_value = result
    store.get_lesson_link.return_value = None
    return store


class TestCourseSearchToolExecute:
    def setup_method(self):
        self.mock_store = make_mock_store()
        self.tool = CourseSearchTool(self.mock_store)

    # ------------------------------------------------------------------
    # Happy-path: results found
    # ------------------------------------------------------------------

    def test_execute_returns_formatted_content_when_results_exist(self):
        """execute() returns the document text when the vector store has matches."""
        self.mock_store.search.return_value = SearchResults(
            documents=["MCP lets models call external tools."],
            metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
            distances=[0.1],
        )
        self.mock_store.get_lesson_link.return_value = "https://example.com/lesson1"

        result = self.tool.execute(query="what is MCP")

        assert "MCP lets models call external tools." in result
        assert "MCP Course" in result

    def test_execute_includes_lesson_number_in_header(self):
        """When a lesson_number is in metadata, the header includes 'Lesson N'."""
        self.mock_store.search.return_value = SearchResults(
            documents=["Some content"],
            metadata=[{"course_title": "AI Basics", "lesson_number": 3}],
            distances=[0.2],
        )

        result = self.tool.execute(query="AI")

        assert "Lesson 3" in result

    # ------------------------------------------------------------------
    # Empty results
    # ------------------------------------------------------------------

    def test_execute_returns_no_content_message_when_empty(self):
        """execute() returns 'No relevant content found' when store is empty."""
        # SearchResults with empty lists already set by setup_method
        result = self.tool.execute(query="obscure topic nobody wrote about")

        assert "No relevant content found" in result

    def test_execute_includes_course_filter_in_empty_message(self):
        """Empty-result message includes the course_name filter when provided."""
        result = self.tool.execute(query="topic", course_name="MCP")

        assert "No relevant content found" in result
        assert "MCP" in result

    def test_execute_includes_lesson_filter_in_empty_message(self):
        """Empty-result message includes the lesson_number filter when provided."""
        result = self.tool.execute(query="topic", lesson_number=5)

        assert "No relevant content found" in result
        assert "5" in result

    # ------------------------------------------------------------------
    # Parameter forwarding
    # ------------------------------------------------------------------

    def test_execute_passes_query_to_vector_store(self):
        """execute() forwards the query string to VectorStore.search()."""
        self.tool.execute(query="transformer architecture")

        self.mock_store.search.assert_called_once()
        call_kwargs = self.mock_store.search.call_args[1]
        assert call_kwargs["query"] == "transformer architecture"

    def test_execute_passes_course_name_to_vector_store(self):
        """execute() forwards course_name to VectorStore.search()."""
        self.tool.execute(query="q", course_name="Intro to ML")

        call_kwargs = self.mock_store.search.call_args[1]
        assert call_kwargs["course_name"] == "Intro to ML"

    def test_execute_passes_lesson_number_to_vector_store(self):
        """execute() forwards lesson_number to VectorStore.search()."""
        self.tool.execute(query="q", lesson_number=7)

        call_kwargs = self.mock_store.search.call_args[1]
        assert call_kwargs["lesson_number"] == 7

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_execute_propagates_vector_store_error(self):
        """When SearchResults.error is set, execute() returns the error string."""
        self.mock_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="ChromaDB connection failed",
        )

        result = self.tool.execute(query="anything")

        assert "ChromaDB connection failed" in result

    # ------------------------------------------------------------------
    # Source tracking
    # ------------------------------------------------------------------

    def test_execute_populates_last_sources_with_label_and_url(self):
        """After a successful search, last_sources is populated with label and url."""
        self.mock_store.search.return_value = SearchResults(
            documents=["content"],
            metadata=[{"course_title": "AI Course", "lesson_number": 2}],
            distances=[0.2],
        )
        self.mock_store.get_lesson_link.return_value = "https://example.com/lesson2"

        self.tool.execute(query="AI")

        assert len(self.tool.last_sources) == 1
        src = self.tool.last_sources[0]
        assert src["label"] == "AI Course - Lesson 2"
        assert src["url"] == "https://example.com/lesson2"

    def test_execute_clears_previous_sources_on_new_search(self):
        """last_sources from a previous search should be replaced, not appended."""
        # First search
        self.mock_store.search.return_value = SearchResults(
            documents=["first result"],
            metadata=[{"course_title": "Course A", "lesson_number": 1}],
            distances=[0.1],
        )
        self.tool.execute(query="first query")
        assert len(self.tool.last_sources) == 1

        # Second search returns 2 results
        self.mock_store.search.return_value = SearchResults(
            documents=["r1", "r2"],
            metadata=[
                {"course_title": "Course B", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
        )
        self.tool.execute(query="second query")

        assert len(self.tool.last_sources) == 2

    # ------------------------------------------------------------------
    # Regression: MAX_RESULTS = 0 symptom
    # ------------------------------------------------------------------

    def test_zero_results_from_store_always_yields_no_content_message(self):
        """
        Simulates what happens when config.MAX_RESULTS = 0.

        ChromaDB returns an empty result set, so execute() always reports
        'No relevant content found'. This is the symptom the user sees as
        'query failed' answers.
        """
        # Store returns nothing (as it would with n_results=0 in ChromaDB)
        self.mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = self.tool.execute(query="What is a transformer model?")

        assert "No relevant content found" in result, (
            "With MAX_RESULTS=0, ChromaDB returns zero documents. "
            "The chatbot will always answer 'No relevant content found' "
            "for content questions, appearing as 'query failed'."
        )


class TestToolManager:
    def test_register_and_retrieve_tool_definition(self):
        """Registered tools appear in get_tool_definitions()."""
        manager = ToolManager()
        mock_store = make_mock_store()
        tool = CourseSearchTool(mock_store)
        manager.register_tool(tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_execute_tool_delegates_to_registered_tool(self):
        """execute_tool() calls the registered tool's execute()."""
        manager = ToolManager()
        mock_store = make_mock_store(
            documents=["lesson text"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
        )
        tool = CourseSearchTool(mock_store)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="test")

        assert "lesson text" in result

    def test_execute_unknown_tool_returns_error_message(self):
        """execute_tool() with unknown name returns an error string."""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "not found" in result.lower()

    def test_get_last_sources_returns_sources_from_search_tool(self):
        """get_last_sources() picks up last_sources from CourseSearchTool."""
        manager = ToolManager()
        mock_store = make_mock_store(
            documents=["content"],
            metadata=[{"course_title": "C", "lesson_number": 1}],
            distances=[0.1],
        )
        tool = CourseSearchTool(mock_store)
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="q")

        sources = manager.get_last_sources()

        assert len(sources) == 1

    def test_reset_sources_clears_last_sources(self):
        """reset_sources() empties last_sources on every registered tool."""
        manager = ToolManager()
        mock_store = make_mock_store(
            documents=["content"],
            metadata=[{"course_title": "C", "lesson_number": 1}],
            distances=[0.1],
        )
        tool = CourseSearchTool(mock_store)
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="q")
        assert len(tool.last_sources) == 1  # sanity check

        manager.reset_sources()

        assert tool.last_sources == []
