"""
Meant to be used alongside chromadb, but any vectordb that uses KNN will work too.
Tunes sequence embeddings to allow the game to more accurately predict player's intentions.
"""
import numpy
import json
from typing import Callable
from langchain.schema import BaseRetriever
from langchain.embeddings.base import Embeddings # TODO: Make all embedding_function Embeddings interface

class EdgeEmbedding:
    """Edge embedding dictionary that stores, calculate and allows for retrieval of different preset embeddings"""
    def __init__(self, name: str, dims: int, embedding_function: Callable[[list], list]=None, current_embedding: list=None, current_weight: int=0) -> None:
        """
        Constructor. If loading from dict, use from_dict() instead. 
        Params:
            - name: identifier
            - dims: number of dimensions. e.g. 28 emotions -> 28-d embedding TODO: is this necessary? currently here for consistency
            - embedding_function: the function that takes a batch of responses and returns the corresponding batch of embeddings. If not provided, the embedding is marked as Final (no tuning allowed).
            - current_embedding: the current embedding loaded from json, or pre-determined to bypass tuning. Defaulted to None.
            - current_weight: the current weight loaded from json (use 1 to bypass tuning). If current_weight is 0, the current_embedding will be ignored. Defaulted to 0.
        """
        if not name:
            raise Exception("edge embedding name cannot be None")

        self.name = name
        self._final = False # Whether tuning is disabled on this embedding

        if (embedding_function is not None):
            # Validating embedding function
            test_embeddings = embedding_function(["test string"]) # This must not throw an error
            test_dims = len(test_embeddings[0])
            if test_dims != dims:
                raise Exception("Embedding function is not valid. Returns embedding with {} dims instead of the defined {} dims".format(test_dims, dims))

            self.embed = embedding_function
        else:
            self._final = True

        if not current_embedding:
            self.embedding = [0]*dims
        else:
            if len(current_embedding) != dims: # Validates embedding
                raise Exception("Embedding is not valid. Embedding has {} dims instead of the defined {} dims".format(test_dims, dims))

            self.embedding = current_embedding

        self.weight = current_weight # The number of responses this embedding represents
        self.n_dims = dims

    @staticmethod
    def from_dict(embedding_dict: dict, embedding_function: Callable[[list], list]=None) -> 'EdgeEmbedding':
        """
        Generate up a edge Embedding instance. DO NOT USE TO MAKE DEEP COPY, use copy() instead
        Params:
            - embedding_dict: the result of edgeEmbedding.to_dict() instance method, or a dict with the same format
            - embedding_function: the function that takes a batch of responses and returns the corresponding batch of embeddings. Set to None if no tuning is required
        """
        name = embedding_dict["name"]
        embedding = embedding_dict["embedding"]
        dims = len(embedding)
        weight = embedding_dict["weight"]
        return EdgeEmbedding(name, dims, embedding_function, embedding, weight)

    def get_embedding(self) -> list:
        """
        Getter function for embedding attribute. Defaulted to all 0s if no responses have been added.
        """
        return self.embedding

    def tune_prompts(self, prompts: list) -> list:
        """
        Add prompts to the embedding and calculates new embedding using weighted average. Tunes embedding of SequenceEvent.
        Potentially suffers from density bias. Good response choices will depend on choice of embedding function.
        Params:
            -  prompts: the list of prompts that should be matched to this edge_embedding instance

        Returns: The new embedding
        """
        if self._final:
            print("The Embedding is marked as Final. Cannot be tuned.")
            return None

        new_embeddings = self.embed(prompts)

        # Update weighted average
        self.embedding = numpy.average([self.embedding] + new_embeddings, 0, [self.weight]+[1]*len(prompts)).tolist()

        # Update weights
        self.weight += len(prompts)
        return self.embedding

    def to_dict(self) -> dict:
        """
        Used to save to json as part of configuration, as well as make a copy
        Returns: a dict of the form {'name': ..., 'embedding': ..., 'weight': ...} 
        """
        return {
            'name': self.name,
            'embedding': self.embedding,
            'weight': self.weight
        }

    def copy(self, new_name: str) -> 'EdgeEmbedding':
        """
        Return a deep copy of this edgeEmbedding instance.
        Params:
            - new_name: Give new name. Ensure it is unique in the game to avoid unexpected behaviours.
        """
        embedding_dict = self.to_dict()
        embedding_dict["name"] = new_name
        return self.from_dict(embedding_dict, self.embed)


class GameNode:
    """Wrapper for a document in Database"""
    def __init__(self, id: str, context: str, metadata: str) -> None:
        """
        Constructor. To build the event from dict, use .from_dict()
        Params:
            - id: self-explanatory
            - context: also self-explanatory
            - metadata: the text prompts given as response to player speech.
        """
        self.id = id
        self.context = context
        self.metadata = metadata # Future proofing

    def to_dict(self) -> dict:
        """Saves tree to a dictionary that can be serialized with json"""
        return {
            'id': self.id,
            'context': self.context,
            'metadata': self.metadata
        }

    @staticmethod
    def from_dict(node_dict: dict) -> 'GameNode':
        """Rebuild tree using a dictionary to resume progress (expects json deserialization higher up in the process)"""
        return GameNode(node_dict["id"], node_dict["context"], node_dict["metadata"])


class BaseGameTree:
    """
    Provides methods that supports building out a Game Tree and acts as an interface for Database.
    Only Constructor can exit to support retries.
    Technically a graph, not a tree.
    """
    def __init__(self, name: str, dims: int, embedding_function: Callable[[list], list]=None, ) -> None:
        """
        Constructor.
        Params:
            - name: Name of the story (Must be chromadb friendly)
            - embedding_function: Takes input and returns embedding. If not provided, the Tree is marked as Final (cannot be changed)
        """
        self.name = name
        self._final = False # Final flag, determines if any change can be made to the tree
        # if flag is False, embedding_function will be used to determine embedding using document text

        if embedding_function is not None:
            # Validating embedding function
            test_embeddings = embedding_function(["test string"]) # This must not throw an error
            test_dims = len(test_embeddings[0])
            if test_dims != dims:
                raise Exception("Embedding function is not valid. Returns embedding with {} dims instead of the defined {} dims".format(test_dims, dims))

            self.embed = embedding_function
        else:
            self._final = True

        self.dims = dims

        # Currently, if there are 2 ways to reach an event, a copy with a new unique id must be made
        self.node_dict = {} # id - SequenceEvent. Mostly for lookup

        # Stores edgeEmbedding templates, must be generated first, tuned later
        default_embedding = EdgeEmbedding("default", dims, embedding_function) # All 0, must be tuned
        self.embedding_template_dict = {"default": default_embedding} # Must be copied using .copy() to be used or risk unexpected behaviour

        # Provides edge look up for custom edge embedding. Format: {(SequenceEvent left, SequenceEvent right): edgeEmbedding embedding}
        # Where left is start event, right is end event, and embedding is the required embedding to go from left to right
        self.edge_dict = {} 

    def add_node(self, node: GameNode) -> bool:
        """
        Add a GameNode to the GameTree
        Params:
            - node: a GameNode instance
        """
        if self._final:
            print("Game Tree was marked as Final. No change can be made to it")
            return False

        event_id = node.id

        if event_id in self.node_dict:
            print("Sequence Event id {} already exists. If there is a second way to reach this event, create a copy with a unique id and retry")
            return False
        else:
            self.node_dict[event_id] = node
            return True

    def add_embedding_template(self, edge_embedding: EdgeEmbedding) -> bool:
        """
        Add embedding template.
        Params:
            - name: the name for the template
            - prompts: the initial prompts to tune the template. If None, consider using the "default" template instead.
        """
        if self._final:
            print("Game Tree was marked as Final. No change can be made to it")
            return False

        name = edge_embedding.name

        if name in self.embedding_template_dict:
            print("Embedding template {} already exists. Use a new unique name")
            return False
        else:
            self.embedding_template_dict[name] = edge_embedding
            return True

    def add_edge(self, start_id: str, end_id: str, embedding_name: str, embedding_template: str="default") -> bool:
        """
        Add edge to the tree and assign a premade embedding template. 
        Most of the time, you only have to override validate_edge() to modify behaviour.
        Params:
            - start_id: The id of the start event. Must exists in event_dict.
            - end_id: The id of the end event. Must exists in event_dict.
            - embedding_name: Rename the embedding class. Ensure it is unique to avoid unexpected behaviours.
            - embedding_template: Name of embedding template to use. Must exists in embedding_template_dict.
        """
        if self.validate_edge(start_id, end_id, embedding_name, embedding_template):
            self.edge_dict[(start_id, end_id)] = self.embedding_template_dict[embedding_template].copy(embedding_name)
            return True

    def validate_edge(self, start_id: str, end_id: str, embedding_name: str, embedding_template: str="default") -> bool:
        """Validates new edge_dict entry. Only contains basic validation, should be overidden to prevent undesireable behaviour"""
        if self._final:
            print("Game Tree was marked as Final. No change can be made to it")
            return False
        elif embedding_template not in self.embedding_template_dict:
            print("Invalid template: Embedding Template must be one of {}. Create one with add_embedding_template() or use the default template".format(self.embedding_template_dict.keys()))
            return False
        elif (start_id, end_id) in self.edge_dict:
            print("Invalid edge: edge {} already exists".format(start_id + '-' + end_id))
            return False
        else:
            return True

    def tune_edge(self, edge_id: tuple, prompts: list) -> bool:
        """
        Tune the chosen embedding with extra prompts.
        Params:
            - edge_id: Identifier for edge, is a tuple of form (start_event, end_event)
            - prompts: List of prompts for tuning
        """
        if self._final:
            print("Game Tree was marked as Final. No change can be made to it")
            return False

        if edge_id not in self.edge_dict:
            print("edge {} does not exists".format(edge_id))
            return False
        self.edge_dict[edge_id].tune_prompts(prompts)
        return True

    def get_template_options(self) -> list:
        """Get the name of all stored templates"""
        return self.embedding_template_dict.keys()

    @staticmethod
    def build_from_json(edge_to_json: str, embedding_function: Callable[[list], list]=None) -> 'BaseGameTree':
        """
        Rebuild tree from JSON file. May cause unexpected behaviour if the JSON file was built using a derived Tree.
        Params:
            - edge_to_json: The string that signifies the edge to the jsonified tree
            - embedding_function: Takes input and returns embedding. If not provided, the Tree is marked as Final (cannot be changed)
        """
        tree_dict = {}
        with open(edge_to_json, "r") as f:
            tree_dict = json.load(f)

        builder = BaseGameTree(tree_dict["name"], tree_dict["dims"], embedding_function)
        for (template_name, template_dict) in tree_dict["embedding_template_dict"].items():
            builder.embedding_template_dict[template_name] = EdgeEmbedding.from_dict(template_dict, embedding_function)

        for (event_id, event_dict) in tree_dict["event_dict"].items():
            builder.node_dict[event_id] = GameNode.from_dict(event_dict)

        for (edge_string, embedding_dict) in tree_dict["edge_dict"].items():
            edge = edge_string.split(" ") # Split by " "
            builder.edge_dict[edge] = EdgeEmbedding.from_dict(embedding_dict, embedding_function)
        
        return builder

    def to_dict(self) -> dict:
        """Return the instance in dictionary form to be saved to json"""
        return {
            "name": self.name,
            "embedding_template_dict": {template_id: template.to_dict() for (template_id, template) in self.embedding_template_dict.items()}, # Kinda optional here
            "event_dict": {event_id: event.to_dict() for (event_id, event) in self.node_dict.items()},
            "edge_dict": {edge[0]+" "+edge[1]: embedding.to_dict() for (edge, embedding) in self.edge_dict.items()},
            "dims": self.dims
        }

    def to_json(self, edge_to_json: str) -> None:
        """Saves current Tree configuration to json"""
        with open(edge_to_json, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def write_db(self, chroma_client):
        """Write current tree to a ChromaDB collection. TODO: Make this work with any vectorstore using langchain"""
        collection = chroma_client.get_collection(name=self.name)
        # Only need to add the following nodes to chroma, as the starting state does not need to be defined
        embeddings = []
        documents = []
        metadatas=[]
        ids=[]

        for (edge, embedding) in self.edge_dict.items():
            target_event_name = edge[1]
            target_event = self.node_dict[target_event_name]

            embeddings.append(embedding.get_embedding())
            documents.append(target_event.name)
            ids.append(target_event_name)
            metadatas.append({
                "reachableSequences": json.dumps(target_event.reachable),
                "reaction": target_event.context # TODO Must change later
            })

        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)


class GameNodeRetriever(BaseRetriever):
    """
    Retriever for events in the Game Tree. Requires ChromaDB as langchain vectorstore does not support custom embedding
    Allows Brain to interact with the vectorstore via the langchain BaseRetriever interface
    """
    def __init__(self, tree: BaseGameTree, chroma_client) -> None:
        """
        Constructor.
        Parameters:
            - tree: a GameTree instance that retrieval will be done on
            - chroma_client: chromadb client object
        """
        self._game_tree = tree
        

    def get_relevant_documents(self, query: str) -> list[dict]:
        pass