"""Defines the current game state, including player state, character states as well as world state"""
class Context:
    """
    Interface class for contextual embeddings used to interface with VectorDB.
    Part of Sequence Tree, sent to Brain for processing there, or bypass Brain altogether.
    """
    def __init__(self) -> None:
        pass 

class State:
    """Interface class to define states. Keeps it consistent among different components."""
    def __init__(self) -> None:
        pass