from pathlib import Path
from typing import List, Optional, Type

from pydantic import BaseModel, Field

from composio.tools.local.base import Action
from composio.tools.local.base.utils.grep_utils import grep_util


class CodeSearchRequest(BaseModel):
    search_pattern: str = Field(
        ..., description="Pattern to search for in the codebase"
    )
    search_directory: Optional[str] = Field(
        default=str(Path.home()),
        description="Directory to search in",
        examples=["/user/home"],
    )
    target_files: Optional[List[str]] = Field(
        default=None,
        description="List of specific files to search",
        examples=[
            ["/user/home/file1.py", "/user/home/file2.py"],
            ["/user/home/python/file4.py"],
        ],
    )
    respect_gitignore: bool = Field(
        default=True,
        description="If true, respect .gitignore file and exclude files/directories listed in it from the search",
    )


class SearchResult(BaseModel):
    file_path: str
    matched_content: str


class CodeSearchResponse(BaseModel):
    search_results: List[SearchResult] = Field(..., description="Code search results")
    error_message: Optional[str] = Field(
        default=None, description="Error message if any"
    )


class SearchCodebase(Action[CodeSearchRequest, CodeSearchResponse]):
    """
    Searches the codebase for a specific pattern using grep-like functionality.

    Example:
    ```
        search_pattern="def main",
        search_directory="/path/to/project",
        case_sensitive=True,
        respect_gitignore=True
    ```
    """

    _display_name = "Search Codebase"
    _description = "Regex Searches the codebase for a specific pattern in an optimised fashion, similar to the grep command."
    _request_schema: Type[CodeSearchRequest] = CodeSearchRequest
    _response_schema: Type[CodeSearchResponse] = CodeSearchResponse
    _tags = ["search"]
    _tool_name = "codesearch"

    def execute(
        self, request: CodeSearchRequest, authorisation_data: dict = {}
    ) -> CodeSearchResponse:
        try:
            if request.target_files:
                search_paths = request.target_files
            elif request.search_directory:
                search_paths = [request.search_directory]
            else:
                search_paths = [str(Path.home())]

            grep_results = grep_util(
                pattern=request.search_pattern,
                filenames=search_paths,
                no_gitignore=not request.respect_gitignore,
            )

            formatted_results = [
                SearchResult(
                    file_path=result["filename"], matched_content=result["matches"]
                )
                for result in grep_results
            ]

            return CodeSearchResponse(search_results=formatted_results)
        except Exception as e:
            error_message = f"An error occurred during code search: {str(e)}"
            print(error_message)
            return CodeSearchResponse(search_results=[], error_message=error_message)
