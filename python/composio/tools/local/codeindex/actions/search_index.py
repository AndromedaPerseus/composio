import os
from pathlib import Path
from typing import Type, List, Optional
from pydantic import BaseModel, Field

from composio.tools.local.base import Action
from composio.tools.local.codeindex.actions.create_index import (
    CreateIndex,
    DEFAULT_EMBEDDING_MODEL_LOCAL,
    DEFAULT_EMBEDDING_MODEL_REMOTE,
    SUPPORTED_FILE_EXTENSIONS,
)


class SearchCodebaseRequest(BaseModel):
    directory: str = Field(..., description="Directory to search")
    query: str = Field(..., description="Search query")
    result_count: int = Field(default=5, description="Number of results to return")
    file_type: str = Field(
        None,
        description="File type to filter results (case-insensitive). Supported types: PY, JS, TS, HTML, CSS, JAVA, C++, C, CHeader, MD, TXT",
    )


class SearchResult(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    content: str
    similarity: float
    file_type: str


class SearchCodebaseResponse(BaseModel):
    results: List[SearchResult] = Field(..., description="Search results")
    error: Optional[str] = Field(default=None, description="Error message if any")


class SearchCodebase(Action[SearchCodebaseRequest, SearchCodebaseResponse]):
    """
    Searches the indexed code base for relevant code snippets.
    """

    _display_name = "Search Codebase"
    _description = "Searches the indexed codebase for relevant code snippets."
    _request_schema: Type[SearchCodebaseRequest] = SearchCodebaseRequest
    _response_schema: Type[SearchCodebaseResponse] = SearchCodebaseResponse
    _tags = ["index", "search"]
    _tool_name = "codeindex"

    def execute(
        self, request: SearchCodebaseRequest, authorisation_data: dict = {}
    ) -> SearchCodebaseResponse:
        import chromadb
        from chromadb.errors import ChromaError

        # Check if index exists
        create_index = CreateIndex()
        status = create_index.check_status(request.directory)
        if status["status"] != "completed":
            return SearchCodebaseResponse(
                results=[], error="Index not completed or not found"
            )

        # Set up Chroma client and collection
        index_storage_path = Path.home() / ".composio" / "index_storage"
        chroma_client = chromadb.PersistentClient(path=str(index_storage_path))
        collection_name = Path(request.directory).name

        embedding_type = status.get("embedding_type", "local")
        embedding_function = create_index._create_embedding_function(
            embedding_type,
        )

        try:
            chroma_collection = chroma_client.get_collection(
                name=collection_name, embedding_function=embedding_function
            )

            # Prepare filter based on file_type if provided
            filter_condition = None
            if request.file_type:
                filter_condition = {"file_type": {"$eq": request.file_type.upper()}}

            # Perform the search
            results = chroma_collection.query(
                query_texts=[request.query],
                n_results=request.result_count,
                where=filter_condition,
            )

            # Process and format the results
            search_results = []
            if results["documents"] and results["metadatas"] and results["distances"]:
                for document, metadata, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    search_results.append(
                        SearchResult(
                            file_path=str(metadata["file_path"]),
                            start_line=int(metadata["start_line"]),
                            end_line=int(metadata["end_line"]),
                            content=document,
                            similarity=round(1 - distance, 4),
                            file_type=str(metadata["file_type"]),
                        )
                    )

            return SearchCodebaseResponse(results=search_results)
        except ChromaError as e:
            return SearchCodebaseResponse(
                results=[], error=f"Collection '{collection_name}' not found: {str(e)}"
            )
        except Exception as e:
            error_message = f"An error occurred during search: {str(e)}"
            print(error_message)
            return SearchCodebaseResponse(results=[], error=error_message)
