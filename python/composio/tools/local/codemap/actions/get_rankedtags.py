from pathlib import Path
from typing import List, Optional, Type

from composio.tools.local.base import Action
from composio.tools.local.base.utils.grep_utils import get_files_excluding_gitignore
from composio.tools.local.base.utils.repomap import RepoMap
from pydantic import BaseModel, Field


class GenerateRankedTagsRequest(BaseModel):
    repository_root: str = Field(
        ...,
        description="Absolute path to the root directory of the repository",
        examples=[
            "/home/user/projects/my-repo",
            "C:\\Users\\user\\Documents\\GitHub\\project",
        ],
    )
    files_of_interest: List[str] = Field(
        ...,
        description="List of file paths (relative to the repository root) for which to generate ranked tags",
        examples=[
            ["src/main.py", "tests/test_utils.py"],
            ["lib/core.js", "lib/helpers.js", "src/index.js"],
        ],
    )


class GenerateRankedTagsResponse(BaseModel):
    ranked_tags: str = Field(
        ...,
        description="List of ranked tags for the specified files, ordered by importance",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if any issues occurred during tag generation",
    )


class GenerateRankedTags(Action[GenerateRankedTagsRequest, GenerateRankedTagsResponse]):
    """
    Generates ranked tags for specified files of interest within a repository.

    This action analyzes the repository structure and content to produce a list of
    important code elements (tags) from the specified files, ranked by their
    significance within the codebase.

    Use cases:
    1. Quickly understand the structure and key components of specific files
    2. Identify important functions, classes, or variables in files of interest
    3. Assist in code navigation and comprehension for large codebases

    Example usage:
    ```python
    request =  repository_root="/path/to/repo",
        files_of_interest=["src/main.py", "src/utils.py"]
    )
    result = GenerateRankedTags().execute(request)

    if result.error:
        print(f"Error: {result.error}")
    else:
        for tag in result.ranked_tags:
            print(f"{tag.file_path}:{tag.line_number} - {tag.tag_content}")
    ```
    """

    _display_name = "Generate Ranked Tags"
    _description = "Analyzes specified files in a repository and generates a ranked list of important code elements (tags), providing a structured view of key components."
    _request_schema: Type[GenerateRankedTagsRequest] = GenerateRankedTagsRequest
    _response_schema: Type[GenerateRankedTagsResponse] = GenerateRankedTagsResponse
    _tags = ["repo", "tags", "code-analysis"]
    _tool_name = "codemap"

    def execute(
        self, request: GenerateRankedTagsRequest, authorisation_data: dict = {}
    ) -> GenerateRankedTagsResponse:
        repo_root = Path(request.repository_root).resolve()

        if not repo_root.exists():
            return GenerateRankedTagsResponse(
                ranked_tags="", error=f"Repository root path {repo_root} does not exist"
            )

        try:
            # Get all files in the repository, excluding those in .gitignore
            all_files = get_files_excluding_gitignore(repo_root)

            # Convert absolute paths to relative paths
            all_files = [str(Path(file).relative_to(repo_root)) for file in all_files]

            # Generate ranked tags map
            repo_map = RepoMap(root=repo_root)
            ranked_tags_map = repo_map.get_ranked_tags_map(
                chat_fnames=[],
                other_fnames=all_files,
                mentioned_fnames=set(request.files_of_interest),
                mentioned_idents=set(),
            )

            # Parse the ranked_tags_map string to extract RankedTag objects
            if ranked_tags_map is None:
                return GenerateRankedTagsResponse(
                    ranked_tags="",
                    error="No ranked tags map generated",
                )
            return GenerateRankedTagsResponse(
                ranked_tags=ranked_tags_map,
            )

        except Exception as e:
            return GenerateRankedTagsResponse(
                ranked_tags="",
                error=f"An error occurred while generating ranked tags: {str(e)}",
            )
