"""
Inherits from BaseGameTree. Demonstrates a GameTree for knowledge-based games
"""
import uuid

from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

from lifelike.StateManager.base_game_tree import BaseGameTree

class KnowledgeTree(BaseGameTree):
    """Allows for loading the tree from a pre-defined list of contextual texts"""
    @classmethod
    def from_texts(cls: BaseGameTree, name:str, texts: list[str], embedding_function: Embeddings, metadatas: list[dict]=None, ids: list[str]=None):
        """For now, must know the embedding dimension."""
        if ids is None: # If id not provided, generate it
            ids = [str(uuid.uuid1()) for _ in texts]

        tree = cls(name, embedding_function)
        # TODO: Currently bypasses building tree. Tree is essentially non-functional
        tree.vectorstore = Chroma.from_texts(texts, embedding_function, metadatas, ids)