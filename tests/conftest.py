"""
Root conftest: patch langgraph sub-modules with MagicMocks so that
langchain==1.2.0 imports succeed even though langgraph==1.0.5 does not
expose the expected symbols.

This shim has no effect on test correctness.
"""
import sys
from unittest.mock import MagicMock


def _mock_module(name: str) -> MagicMock:
    mod = MagicMock()
    mod.__name__ = name
    mod.__loader__ = None
    mod.__package__ = name.rsplit(".", 1)[0]
    mod.__spec__ = None
    sys.modules[name] = mod
    return mod


# Stub the langgraph sub-modules that langchain==1.2.0 requires but
# langgraph==1.0.5 does not provide.
_mock_module("langgraph.prebuilt.tool_node")
# Also patch langgraph.prebuilt itself so attribute access works
import langgraph.prebuilt as _lgp  # noqa: E402 — already installed

_lgp.InjectedState = MagicMock()
_lgp.InjectedStore = MagicMock()
_lgp.ToolRuntime = MagicMock()
