import os
from pathlib import Path
from typing import Type, List, Optional
from pydantic import BaseModel, Field
from composio.tools.local.base.utils.repomap import RepoMap
from composio.tools.local.base import Action
from composio.tools.local.base.utils.grep_utils import get_files_excluding_gitignore


class GenerateRankedTagsRequest(BaseModel):
    root_path: str = Field(..., description="Root path of the repository")
    target_files: List[str] = Field(
        ..., description="Files of particular interest to generate ranked tags for"
    )


class RankedTag(BaseModel):
    file_path: str
    line_number: int
    tag_content: str


class GenerateRankedTagsResponse(BaseModel):
    ranked_tags: List[RankedTag] = Field(
        ..., description="Ranked tags for the specified files"
    )
    error: Optional[str] = Field(default=None, description="Error message if any")


class GenerateRankedTags(Action[GenerateRankedTagsRequest, GenerateRankedTagsResponse]):
    """
    Generates ranked tags for files of particular interest in the repository.
    """

    _display_name = "Generate Ranked Tags"
    _description = "Generates ranked tags for files of particular interest in the repository, providing a structured view of important code elements."
    _request_schema: Type[GenerateRankedTagsRequest] = GenerateRankedTagsRequest
    _response_schema: Type[GenerateRankedTagsResponse] = GenerateRankedTagsResponse
    _tags = ["repo", "tags"]
    _tool_name = "codemap"

    def execute(
        self, request: GenerateRankedTagsRequest, authorisation_data: dict = {}
    ) -> dict:

        repo_root = Path(request.root_path).resolve()

        if not repo_root.exists():
            return {"error": f"Repository root path {repo_root} does not exist"}

        try:
            # Get all files in the repository, excluding those in .gitignore
            all_files = get_files_excluding_gitignore(repo_root)

            # Convert absolute paths to relative paths
            all_files = [str(Path(file).relative_to(repo_root)) for file in all_files]

            # Generate ranked tags map
            ranked_tags = RepoMap(root=repo_root).get_ranked_tags_map(
                chat_fnames=[],
                other_fnames=all_files,
                mentioned_fnames=set(request.target_files),
                mentioned_idents=set(),
            )

            # Convert the ranked tags to a list of RankedTag objects
            ranked_tag_objects = []
            if ranked_tags is None:
                return {"error": "No ranked tags found"}

            for tag in ranked_tags:
                if isinstance(tag, tuple) and len(tag) >= 4:
                    ranked_tag_objects.append(
                        RankedTag(
                            file_path=tag[0],
                            line_number=int(tag[3]),
                            tag_content=tag[2],
                        )
                    )

            return {
                "ranked_tags": ranked_tag_objects,
                "message": "Ranked tags for specified files generated successfully",
            }

        except Exception as e:
            return {
                "error": f"An error occurred while generating ranked tags: {str(e)}"
            }
