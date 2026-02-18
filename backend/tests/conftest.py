import sys
import os

# Add backend/ to path so all backend modules are importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_rag_system():
    """A MagicMock standing in for RAGSystem with realistic defaults."""
    rag = MagicMock()
    rag.session_manager.create_session.return_value = "test-session-id"
    rag.query.return_value = ("Here is the answer.", [])
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"],
    }
    return rag


@pytest.fixture
def mock_config():
    """A MagicMock Config with sensible test defaults."""
    cfg = MagicMock()
    cfg.CHUNK_SIZE = 800
    cfg.CHUNK_OVERLAP = 100
    cfg.MAX_RESULTS = 5
    cfg.MAX_HISTORY = 2
    cfg.CHROMA_PATH = "./test_chroma_db"
    cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    cfg.ANTHROPIC_API_KEY = "test-key"
    cfg.ANTHROPIC_MODEL = "claude-test"
    return cfg
