import os
from pathlib import Path
from typing import Type
from pydantic import BaseModel, Field

from composio.tools.local.base import Action
from composio.tools.local.codeindex.actions.create_index import CreateIndex


class IndexStatusInputSchema(BaseModel):
    dir_to_index_path: str = Field(
        ..., description="Directory to check indexing status"
    )


class IndexStatusOutputSchema(BaseModel):
    status: str = Field(..., description="Status of the indexing process")
    error: str = Field("", description="Error message if indexing failed")


class IndexStatus(Action[IndexStatusInputSchema, IndexStatusOutputSchema]):
    """
    Checks the status of the indexing process for a given directory.
    """

    _display_name = "Check Index Status"
    _description = "Checks the status of the indexing process for a given directory."
    _request_schema: Type[IndexStatusInputSchema] = IndexStatusInputSchema
    _response_schema: Type[IndexStatusOutputSchema] = IndexStatusOutputSchema
    _tags = ["index"]
    _tool_name = "codeindex"

    def execute(
        self, request_data: IndexStatusInputSchema, authorisation_data: dict = {}
    ) -> IndexStatusOutputSchema:
        create_index = CreateIndex()
        status_data = create_index.check_status(request_data.dir_to_index_path)

        return IndexStatusOutputSchema(
            status=status_data.get("status", "unknown"),
            error=status_data.get("error", ""),
        )
