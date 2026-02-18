import anthropic
from typing import List, Optional


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **Search up to two times per query** — use a second search only when the first result is
  insufficient to answer the question (e.g., comparing two courses, multi-part questions)
- Do not search twice for the same information
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    MAX_TOOL_ROUNDS = 2

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to MAX_TOOL_ROUNDS sequential tool-calling rounds.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]
        rounds_completed = 0
        response = None

        while rounds_completed < self.MAX_TOOL_ROUNDS:
            # Build params for this loop iteration (always includes tools)
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
            }
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            response = self.client.messages.create(**api_params)

            # Claude answered directly, or no tool manager wired up — exit loop
            if response.stop_reason != "tool_use" or tool_manager is None:
                break

            # Append Claude's assistant turn (contains the tool_use blocks)
            messages.append({"role": "assistant", "content": response.content})

            # Execute every tool call in this round
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(block.name, **block.input)
                    except Exception as e:
                        result = f"Tool execution failed: {str(e)}"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            # Defensive: stop_reason said tool_use but no tool blocks were present
            if not tool_results:
                break

            # Append tool results as the next user turn
            messages.append({"role": "user", "content": tool_results})
            rounds_completed += 1

        # Fast path: Claude answered directly at some point in the loop
        if response.stop_reason != "tool_use":
            return self._extract_text(response)

        # Slow path: exhausted MAX_TOOL_ROUNDS — final synthesis call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }
        final_response = self.client.messages.create(**final_params)
        return self._extract_text(final_response)

    def _extract_text(self, response) -> str:
        """Return the text of the first text block in the response, or '' if none."""
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""
