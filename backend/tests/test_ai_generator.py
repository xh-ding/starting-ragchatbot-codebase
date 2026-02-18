"""
Tests for AIGenerator.generate_response() in ai_generator.py.

Verifies that the generator:
  - Detects tool_use stop_reason and calls the tool manager
  - Does NOT call tools when Claude answers directly
  - Excludes tools from the second API call (prevents infinite loops)
  - Always includes the system prompt
  - Appends conversation history to the system prompt
"""

import pytest
from unittest.mock import MagicMock, call
from ai_generator import AIGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tool_use_response(tool_name="search_course_content", tool_id="tool_001", tool_input=None):
    """Return a mock Anthropic response that requests a tool call."""
    if tool_input is None:
        tool_input = {"query": "test query"}

    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.id = tool_id
    block.input = tool_input

    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [block]
    return resp


def make_text_response(text="Here is the answer."):
    """Return a mock Anthropic response that contains plain text."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [block]
    return resp


def make_generator():
    """Return an AIGenerator with a mocked Anthropic client."""
    gen = AIGenerator(api_key="test-key", model="claude-test")
    gen.client = MagicMock()
    return gen


# ---------------------------------------------------------------------------
# Tests: tool calling
# ---------------------------------------------------------------------------

class TestAIGeneratorToolCalling:

    def test_tool_manager_called_when_stop_reason_is_tool_use(self):
        """When Claude's stop_reason is 'tool_use', tool_manager.execute_tool() is called."""
        gen = make_generator()
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "search result text"

        first = make_tool_use_response(tool_name="search_course_content",
                                       tool_id="t1",
                                       tool_input={"query": "what is MCP"})
        second = make_text_response("MCP is a protocol.")
        gen.client.messages.create.side_effect = [first, second]

        result = gen.generate_response(
            query="what is MCP",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="what is MCP"
        )
        assert result == "MCP is a protocol."

    def test_tool_manager_not_called_when_stop_reason_is_end_turn(self):
        """When Claude answers directly (end_turn), no tool is executed."""
        gen = make_generator()
        mock_tool_manager = MagicMock()
        gen.client.messages.create.return_value = make_text_response("Direct answer.")

        result = gen.generate_response(
            query="What is Python?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        mock_tool_manager.execute_tool.assert_not_called()
        assert result == "Direct answer."

    def test_second_api_call_excludes_tools(self):
        """
        After executing a tool, the follow-up API call must NOT include 'tools'.
        This prevents Claude from calling tools again in an infinite loop.
        """
        gen = make_generator()
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "search results"

        gen.client.messages.create.side_effect = [
            make_tool_use_response(tool_input={"query": "q"}),
            make_text_response("final"),
        ]

        gen.generate_response(
            query="q",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        second_call_kwargs = gen.client.messages.create.call_args_list[1][1]
        assert "tools" not in second_call_kwargs, (
            "Second API call must not include 'tools' to prevent recursive tool calls."
        )

    def test_exactly_two_api_calls_made_when_tool_used(self):
        """Using a tool should result in exactly two API calls — no more, no less."""
        gen = make_generator()
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "results"

        gen.client.messages.create.side_effect = [
            make_tool_use_response(tool_input={"query": "q"}),
            make_text_response("done"),
        ]

        gen.generate_response(
            query="q",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        assert gen.client.messages.create.call_count == 2

    def test_tool_result_included_in_second_api_call_messages(self):
        """The tool's output must appear in the messages sent to the second API call."""
        gen = make_generator()
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Lesson 1 content about agents."

        gen.client.messages.create.side_effect = [
            make_tool_use_response(tool_id="t99", tool_input={"query": "agents"}),
            make_text_response("answer"),
        ]

        gen.generate_response(
            query="agents",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        second_call_messages = gen.client.messages.create.call_args_list[1][1]["messages"]
        # The last user message should contain the tool_result
        tool_result_messages = [
            m for m in second_call_messages
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(r.get("type") == "tool_result" for r in m["content"])
        ]
        assert len(tool_result_messages) == 1
        tool_result_content = tool_result_messages[0]["content"][0]
        assert tool_result_content["content"] == "Lesson 1 content about agents."
        assert tool_result_content["tool_use_id"] == "t99"


# ---------------------------------------------------------------------------
# Tests: system prompt
# ---------------------------------------------------------------------------

class TestAIGeneratorSystemPrompt:

    def test_system_prompt_is_always_included(self):
        """Every API call must include a non-empty 'system' field."""
        gen = make_generator()
        gen.client.messages.create.return_value = make_text_response()

        gen.generate_response(query="Hello")

        call_kwargs = gen.client.messages.create.call_args[1]
        assert "system" in call_kwargs
        assert len(call_kwargs["system"]) > 0

    def test_conversation_history_appended_to_system_prompt(self):
        """When conversation_history is provided, it appears after SYSTEM_PROMPT."""
        gen = make_generator()
        gen.client.messages.create.return_value = make_text_response()

        gen.generate_response(
            query="follow-up question",
            conversation_history="User: hi\nAssistant: hello",
        )

        call_kwargs = gen.client.messages.create.call_args[1]
        system_content = call_kwargs["system"]
        assert "Previous conversation" in system_content
        assert "User: hi" in system_content
        assert "hello" in system_content

    def test_no_history_uses_base_system_prompt_only(self):
        """When there is no history, 'Previous conversation' is not in system."""
        gen = make_generator()
        gen.client.messages.create.return_value = make_text_response()

        gen.generate_response(query="standalone question")

        call_kwargs = gen.client.messages.create.call_args[1]
        assert "Previous conversation" not in call_kwargs["system"]

    def test_system_prompt_instructs_one_search_per_query(self):
        """The system prompt must contain the one-search-per-query instruction."""
        assert "One search per query" in AIGenerator.SYSTEM_PROMPT or \
               "one search" in AIGenerator.SYSTEM_PROMPT.lower(), (
            "System prompt must instruct Claude to make at most one search per query."
        )


# ---------------------------------------------------------------------------
# Tests: direct (no-tool) response
# ---------------------------------------------------------------------------

class TestAIGeneratorDirectResponse:

    def test_returns_text_directly_when_no_tool_use(self):
        """Without tool use, the response is returned from the first API call."""
        gen = make_generator()
        gen.client.messages.create.return_value = make_text_response("Simple answer.")

        result = gen.generate_response(query="What is 2+2?")

        assert result == "Simple answer."
        assert gen.client.messages.create.call_count == 1

    def test_only_one_api_call_when_no_tool_use(self):
        """Non-tool responses must use exactly one API call."""
        gen = make_generator()
        gen.client.messages.create.return_value = make_text_response("ok")

        gen.generate_response(query="test")

        assert gen.client.messages.create.call_count == 1

    def test_query_is_sent_as_user_message(self):
        """The user query must appear as a user-role message in the API call."""
        gen = make_generator()
        gen.client.messages.create.return_value = make_text_response()

        gen.generate_response(query="my specific question")

        call_kwargs = gen.client.messages.create.call_args[1]
        messages = call_kwargs["messages"]
        user_messages = [m for m in messages if m["role"] == "user"]
        assert any("my specific question" in str(m["content"]) for m in user_messages)
