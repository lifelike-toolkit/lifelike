"""
Inherits from GameNode. Demonstrates a GameTree for knowledge-based games
"""
from langchain.vectorstores import Chroma
from langchain.schema import BaseRetriever

from lifelike.StateManager import BaseGameTree, EdgeEmbedding, GameNode

class TopicNode(GameNode):
    pass

class KnowledgeTree(BaseGameTree):
    pass

class KnowledgeRetriever(BaseRetriever):
    def __init__(self) -> None:
        super().__init__()

    def get_relevant_documents(self, query: str) -> list[dict]:
        return super().get_relevant_documents(query)