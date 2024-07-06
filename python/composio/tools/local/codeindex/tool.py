"""
File I/O tool for Composio.
"""

import typing as t

from composio.tools.local.base import Action, Tool

from .actions import CreateIndex, IndexStatus, SearchIndex


class CodeIndexTool(Tool):
    """Code index tool."""

    def actions(self) -> t.List[t.Type[Action]]:
        """Return the list of actions."""
        return [CreateIndex, IndexStatus, SearchIndex]

    def triggers(self) -> t.List:
        """Return the list of triggers."""
        return []
