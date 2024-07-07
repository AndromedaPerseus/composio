from pathlib import Path
from typing import List, Optional, Type

from pydantic import BaseModel, Field

from composio.tools.local.base import Action
from composio.tools.local.base.utils.grep_utils import get_files_excluding_gitignore
from composio.tools.local.base.utils.repomap import RepoMap


class GetRepoMapRequest(BaseModel):
    root_path: str = Field(..., description="Root path of the repository")
    target_files: List[str] = Field(
        ..., description="Files of particular interest to generate repo map for"
    )


class GetRepoMapResponse(BaseModel):
    repo_map: Optional[str] = Field(
        default=None, description="Generated repository map"
    )
    error: Optional[str] = Field(default=None, description="Error message if any")


class GetRepoMap(Action[GetRepoMapRequest, GetRepoMapResponse]):
    """
    Generates a repository map for specified files of particular interest.
    """

    _display_name = "Get Repository Map"
    _description = "Generates a repository map for files of particular interest, providing a structured view of important code elements in the repository."
    _request_schema: Type[GetRepoMapRequest] = GetRepoMapRequest
    _response_schema: Type[GetRepoMapResponse] = GetRepoMapResponse
    _tags = ["repo"]
    _tool_name = "codemap"

    def execute(
        self, request: GetRepoMapRequest, authorisation_data: dict = {}
    ) -> dict:
        repo_root = Path(request.root_path).resolve()

        if not repo_root.exists():
            return {"error": f"Repository root path {repo_root} does not exist"}

        try:
            # Get all files in the repository, excluding those in .gitignore
            all_files = get_files_excluding_gitignore(repo_root)

            # Convert absolute paths to relative paths
            all_files = [str(Path(file).relative_to(repo_root)) for file in all_files]

            # Generate repo map
            repo_map = RepoMap(root=repo_root).get_repo_map(
                chat_files=[],
                other_files=all_files,
                mentioned_fnames=set(request.target_files),
                mentioned_idents=set(),
            )

            return {
                "repo_map": repo_map,
                "message": "Repository map generated successfully for specified files",
            }

        except Exception as e:
            return {
                "error": f"An error occurred while generating the repository map: {str(e)}"
            }
