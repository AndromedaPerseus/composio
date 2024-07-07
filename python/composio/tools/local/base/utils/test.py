import os
from pathlib import Path

from composio.tools.local.base.utils.repomap import RepoMap

# Test on existing codebase
test_files = [
    "/Users/soham/composio_sdk/python/composio/tools/local/base/utils/file_utils.py",
    "/Users/soham/composio_sdk/python/composio/tools/local/base/utils/parser.py",
    "/Users/soham/composio_sdk/python/composio/tools/local/base/utils/grep_utils.py",
    "/Users/soham/composio_sdk/python/composio/tools/local/base/utils/grep_ast.py",
    "/Users/soham/composio_sdk/python/composio/tools/local/base/utils/file_utils.py",
]

repo_map = RepoMap(root=Path("/Users/soham/composio_sdk"), verbose=False)
other_files = [
    "/Users/soham/composio_sdk/python/composio/tools/local/base/utils/test.py",
    "/Users/soham/composio_sdk/python/composio/tools/local/base/utils/repomap.py",
    "/Users/soham/composio_sdk/python/composio/tools/local/base/utils/file_utils.py",
    "/Users/soham/composio_sdk/python/composio/tools/local/base/utils/parser.py",
    "/Users/soham/composio_sdk/python/composio/tools/local/base/utils/grep_utils.py",
    "/Users/soham/composio_sdk/python/composio/tools/local/base/utils/grep_ast.py",
    "/Users/soham/composio_sdk/python/composio/tools/local/base/utils/file_utils.py",
]
print(
    repo_map.get_repo_map(
        chat_files=test_files,
        other_files=other_files,
        mentioned_idents=["RepoMap"],
    )
)
