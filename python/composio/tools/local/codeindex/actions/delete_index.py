import os
import json
from pathlib import Path
from typing import Type
from pydantic import BaseModel, Field
from composio.tools.local.base import Action
import chromadb
from chromadb.errors import ChromaError


class DeleteIndexInputSchema(BaseModel):
    indexed_dir_to_delete: str = Field(
        ..., description="Directory of the index to delete"
    )


class DeleteIndexOutputSchema(BaseModel):
    result: str = Field(..., description="Result of the action")


class DeleteIndex(Action[DeleteIndexInputSchema, DeleteIndexOutputSchema]):
    """
    Deletes the index for a specified code base.
    """

    _display_name = "Delete index"
    _description = "Deletes the index for a specified code base."
    _request_schema: Type[DeleteIndexInputSchema] = DeleteIndexInputSchema
    _response_schema: Type[DeleteIndexOutputSchema] = DeleteIndexOutputSchema
    _tags = ["index"]
    _tool_name = "codeindex"

    def execute(
        self, request_data: DeleteIndexInputSchema, authorisation_data: dict = {}
    ) -> DeleteIndexOutputSchema:
        index_storage_path = Path.home() / ".composio" / "index_storage"
        collection_name = Path(request_data.indexed_dir_to_delete).name
        status_file = Path(request_data.indexed_dir_to_delete) / ".indexing_status.json"

        try:
            # Delete the collection from Chroma
            chroma_client = chromadb.PersistentClient(path=str(index_storage_path))
            chroma_client.delete_collection(name=collection_name)

            # Delete the status file
            if status_file.exists():
                os.remove(status_file)

            return DeleteIndexOutputSchema(
                result=f"Index for {request_data.indexed_dir_to_delete} has been successfully deleted."
            )
        except ChromaError as e:
            return DeleteIndexOutputSchema(result=f"Failed to delete index: {str(e)}")
        except Exception as e:
            return DeleteIndexOutputSchema(
                result=f"An error occurred while deleting the index: {str(e)}"
            )
