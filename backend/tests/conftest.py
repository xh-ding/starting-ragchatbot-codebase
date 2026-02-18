import sys
import os

# Add backend/ to path so all backend modules are importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
