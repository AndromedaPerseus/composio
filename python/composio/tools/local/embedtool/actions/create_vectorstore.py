from typing import Type

import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from composio.tools.local.base import Action
from llama_index.core import (SimpleDirectoryReader, StorageContext,
                              VectorStoreIndex)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import BaseModel, Field


class VectorStoreInputSchema(BaseModel):
    # Define input schema for your action
    # Example:
    # text: str = Field(..., description="Input text for the action")
    images_path: str = Field(..., description="Path to the saved image folder")
    collection_name: str = Field(..., description="Name of the Chroma VectorStore")
    folder_path: str = Field(..., description="Directory where it should be stored")


class VectorStoreOutputSchema(BaseModel):
    # Define output schema for your action
    # Example:
    result: str = Field(..., description="Result of the action")


class CreateVectorstore(Action[VectorStoreInputSchema, VectorStoreOutputSchema]):
    """
    Creates Vector Store with Image Embeddings
    """

    _display_name = "Create Vector Store"
    _request_schema: Type[VectorStoreInputSchema] = VectorStoreInputSchema
    _response_schema: Type[VectorStoreOutputSchema] = VectorStoreOutputSchema
    _tags = ["store"]
    _tool_name = "embedtool"

    def execute(
        self, request_data: VectorStoreInputSchema, authorisation_data: dict = {}
    ) -> dict:
        # Implement logic to process input and return output
        # Example:
        # response_data = {"result": "Processed text: " + request_data.text}
        embedding_function = OpenCLIPEmbeddingFunction()
        image_loader = ImageLoader()
        # create client and a new collection
        chroma_client = chromadb.PersistentClient(path=request_data.folder_path)
        chroma_collection = chroma_client.create_collection(
            request_data.collection_name,
            embedding_function=embedding_function,
            data_loader=image_loader,
        )

        # load documents
        documents = SimpleDirectoryReader(request_data.images_path).load_data()

        # set up ChromaVectorStore and load in data
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
            storage_context=storage_context,
        )
        index.storage_context.persist(request_data.folder_path)
        return {
            "execution_details": {"executed": True},
            "result": "Vector Store was created with the name:"
            + request_data.collection_name,
        }
