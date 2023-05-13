"""
Inherits from GameNode. Meant for linear games with branching story paths.
Good example of simple GameTree object.
"""
from base_game_tree import BaseGameTree, GameNode, EdgeEmbedding

PathEmbedding = EdgeEmbedding

class SequenceEvent(GameNode):
    """Wrapper for an event document in Database. The game can only see 1 at a time."""
    def __init__(self, id: str, context: str, metadata: dict={}, reachable: list[str]=[]) -> None:
        """
        Constructor. To build the event from dict, use .from_dict()
        Params:
            - id: self-explanatory
            - context: also self-explanatory
            - metadata: the text prompts given as response to player speech.
            - reachable: list of ids for sequence event that this specific event can reach. This behaviour can be customized via SequenceEventRetriever
        """
        super().__init__(id, context, metadata)
        self.metadata["reachable"] = reachable # Internal metadata


class SequenceTree(BaseGameTree):
    """
    Technically a graph. Only used during Database setup. 
    Provides methods that supports building out a sequence tree and acts as an interface for Database.
    Only Constructor can exit to support retries.
    """
    def validate_edge(self, start_id: str, end_id: str, embedding_name: str, embedding_template: str = "default") -> bool:
        """Validates new edge_dict entry. All node in an edge must exist."""
        if not super().validate_edge(start_id, end_id, embedding_name, embedding_template):
            return False
        elif start_id not in self.node_dict or end_id not in self.node_dict:
            print("Either start_id or end_id does not exist in event_dict. Must be 2 of {}.".format(self.node_dict.keys()))
            return False
        else:
            return True

    def add_edge(self, start_id: str, end_id: str, embedding_name: str, embedding_template: str = "default") -> bool:
        if super().add_edge(start_id, end_id, embedding_name, embedding_template):
            self.node_dict[start_id].metadata["reachable"].append(end_id)