"""
Inherits from GameNode. Demonstrates a GameTree for knowledge-based games
"""
from typing import Callable

from langchain.vectorstores import Chroma
from lifelike.StateManager import BaseGameTree

class KnowledgeTree(BaseGameTree):
    @classmethod
    def from_texts(cls: BaseGameTree, name:str, dims:int, texts: list[str], embedding_function: Callable[[list], list], metadatas: list[dict]=None):
        """For now, must know the embedding dimension. TODO intergrate langchain Embedding"""
        tree = cls(name, dims, embedding_function)
        tree.vectorstore = Chroma.from_texts() # Must integrate Embeddings first