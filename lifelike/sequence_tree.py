"""
Meant to be used alongside chromadb, but any vectordb that uses KNN will work too.
Tunes sequence embeddings to allow the game to more accurately predict player's intentions.
"""
import numpy
import json
from typing import Callable

class PathEmbedding:
    """Path embedding dictionary that stores, calculate and allows for retrieval of different preset embeddings"""
    def __init__(self, name: str, dims: int, embedding_function: Callable[[list], list]=None, current_embedding: list=None, current_weight: int=0) -> None:
        """
        Constructor. If loading from dict, use from_dict() instead. 
        Params:
            - name: identifier
            - dims: number of dimensionse. e.g. 28 emotions -> 28-d embedding TODO: is this necessary? currently here for consistency
            - embedding_function: the function that takes a batch of responses and returns the corresponding batch of embeddings. If not provided, the embedding is marked as Final (no tuning allowed).
            - current_embedding: the current embedding loaded from json, or pre-determined to bypass tuning. Defaulted to None.
            - current_weight: the current weight loaded from json (use 1 to bypass tuning). If current_weight is 0, the current_embedding will be ignored. Defaulted to 0.
        """
        if not name:
            raise Exception("Path embedding name cannot be None")

        self.name = name
        self.final = False # Whether tuning is disabled on this embedding

        if (embedding_function is not None):
            # Validating embedding function
            test_embeddings = embedding_function(["test string"]) # This must not throw an error
            test_dims = len(test_embeddings[0])
            if test_dims != dims:
                raise Exception("Embedding function is not valid. Returns embedding with {} dims instead of the defined {} dims".format(test_dims, dims))

            self.embed = embedding_function
        else:
            self.final = True

        if not current_embedding:
            self.embedding = [0]*dims
        else:
            if len(current_embedding) != dims: # Validates embedding
                raise Exception("Embedding is not valid. Embedding has {} dims instead of the defined {} dims".format(test_dims, dims))

            self.embedding = current_embedding

        self.weight = current_weight # The number of responses this embedding represents
        self.n_dims = dims

    @staticmethod
    def from_dict(embedding_dict: dict, embedding_function: Callable[[list], list]=None) -> 'PathEmbedding':
        """
        Generate up a Path Embedding instance. DO NOT USE TO MAKE DEEP COPY, use copy() instead
        Params:
            - embedding_dict: the result of PathEmbedding.to_dict() instance method, or a dict with the same format
            - embedding_function: the function that takes a batch of responses and returns the corresponding batch of embeddings. Set to None if no tuning is required
        """
        name = embedding_dict["name"]
        embedding = embedding_dict["embedding"]
        dims = len(embedding)
        weight = embedding_dict["weight"]
        return PathEmbedding(name, dims, embedding_function, embedding, weight)

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
            -  prompts: the list of prompts that should be matched to this path_embedding instance

        Returns: The new embedding
        """
        if self.final:
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

    def copy(self, new_name: str) -> 'PathEmbedding':
        """
        Return a deep copy of this PathEmbedding instance.
        Params:
            - new_name: Give new name. Ensure it is unique in the game to avoid unexpected behaviours.
        """
        embedding_dict = self.to_dict()
        embedding_dict["name"] = new_name
        return self.from_dict(embedding_dict, self.embed)


class SequenceEvent:
    """Wrapper for an event document in Database. The game can only see 1 at a time."""
    def __init__(self, id: str, name: str, reaction: str, requirements: dict = {}, reachable: list = []) -> None:
        """
        Constructor. To build the event from dict, use .from_dict()
        Params:
            - id: self-explanatory
            - name: also self-explanatory
            - reaction: the text prompts given as response to player speech. WIP. TODO: May want to be its own Context class
            - requirements: a dictionary containing extra requirement, with key-value pair being the dev-defined id of a quantity and its value
            - reachable: list of ids for sequence event that this specific event can reach. Optional, use None to signify game end, or add in later with .add_reachable().
        """
        self.id = id
        self.name = name
        self.context = reaction # Future proofing
        self.requirements = requirements # TODO: Allow for more customization here
        self.reachable = reachable
        # TODO: Process formatted reaction string to allow for a segmented event (Allow player to click on an action button or have another character takeover mid conversation)

    def to_dict(self) -> dict:
        """Saves tree to a dictionary that can be serialized with json"""
        return {
            'id': self.id,
            'name': self.name,
            'context': self.context,
            'requirement': self.requirements,
            'reachable': self.reachable
        }

    @staticmethod
    def from_dict(event_dict: dict) -> 'SequenceEvent':
        """Rebuild tree using a dictionary to resume progress (expects json deserialization higher up in the process)"""
        return SequenceEvent(event_dict["id"], event_dict["name"], event_dict["context"], event_dict["requirement"], event_dict["reachable"])

        
    def add_reachable(self, connecting_event: 'SequenceEvent') -> bool:
        """Adding a reachable event. TODO: handle impossible sequences with custom requirements config"""
        self.reachable.append(connecting_event.id)
        return True



class SequenceTree:
    """
    Technically a graph. Only used during Database setup. 
    Provides methods that supports building out a sequence tree and acts as an interface for Database.
    Only Constructor can exit to support retries.
    """
    def __init__(self, name: str, dims: int, embedding_function: Callable[[list], list]=None) -> None:
        """
        Constructor.
        Params:
            - name: Name of the story (Must be chromadb friendly)
            - embedding_function: Takes input and returns embedding. If not provided, the Tree is marked as Final (cannot be changed)
        """
        self.name = name
        self.final = False

        if embedding_function is not None:
            # Validating embedding function
            test_embeddings = embedding_function(["test string"]) # This must not throw an error
            test_dims = len(test_embeddings[0])
            if test_dims != dims:
                raise Exception("Embedding function is not valid. Returns embedding with {} dims instead of the defined {} dims".format(test_dims, dims))

            self.embed = embedding_function
        else:
            self.final = True

        self.dims = dims

        # Currently, if there are 2 ways to reach an event, a copy with a new unique id must be made
        self.event_dict = {} # id - SequenceEvent. Mostly for lookup

        # Stores PathEmbedding templates, must be generated first, tuned later
        default_embedding = PathEmbedding("default", dims, embedding_function) # All 0, must be tuned
        self.embedding_template_dict = {"default": default_embedding} # Must be copied using .copy() to be used or risk unexpected behaviour

        # Provides path look up. Format: {(SequenceEvent left, SequenceEvent right): PathEmbedding embedding}
        # Where left is start event, right is end event, and embedding is the required embedding to go from left to right
        self.path_dict = {} 

    def add_event(self, event_id: str, name: str, reaction: str, requirements: dict = None) -> bool:
        """
        Add a Sequence Event. Note: The params are close to SequenceEvent's, but reachable will be done in add_path() for consistency.
        Params:
            - id: self-explanatory
            - name: also self-explanatory
            - reaction: the text prompts given as response to player speech. WIP. TODO: May want to be its own Context class
            - requirements: a dictionary containing extra requirement, with key-value pair being the dev-defined id of a quantity and its value
        """
        if self.final:
            print("Sequence Tree was marked as Final. No change can be made to it")
            return False

        if event_id in self.event_dict:
            print("Sequence Event id {} already exists. If there is a second way to reach this event, create a copy with a unique id and retry")
            return False
        else:
            self.event_dict[event_id] = SequenceEvent(event_id, name, reaction, requirements)
            return True

    def add_embedding_template(self, name: str, prompts: list) -> bool:
        """
        Add embedding template.
        Params:
            - name: the name for the template
            - prompts: the initial prompts to tune the template. If None, consider using the "default" template instead.
        """
        if self.final:
            print("Sequence Tree was marked as Final. No change can be made to it")
            return False

        if name in self.embedding_template_dict:
            print("Embedding template {} already exists. Use a new unique name")
            return False
        else:
            embedding_instance = PathEmbedding(name, self.dims, self.embed)
            embedding_instance.tune_prompts(prompts)
            self.embedding_template_dict[name] = embedding_instance
            return True

    def add_path(self, start_id: str, end_id: str, embedding_name: str, embedding_template: str="default") -> bool:
        """
        Add path to the tree and assign a premade embedding template.
        Params:
            - start_id: The id of the start event. Must exists in event_dict.
            - end_id: The id of the end event. Must exists in event_dict.
            - embedding_name: Rename the embedding class. Ensure it is unique to avoid unexpected behaviours.
            - embedding_template: Name of embedding template to use. Must exists in embedding_template_dict.
        """
        if self.final:
            print("Sequence Tree was marked as Final. No change can be made to it")
            return False
        elif start_id not in self.event_dict or end_id not in self.event_dict:
            print("Either start_id or end_id does not exist in event_dict. Must be 2 of {}.".format(self.event_dict.keys()))
            return False
        elif start_id == end_id:
            print("Invalid path: start_id and end_id are both {}.".format(start_id))
            return False
        elif embedding_template not in self.embedding_template_dict:
            print("Invalid template: Embedding Template must be one of {}. Create one with add_embedding_template() or use the default template".format(self.embedding_template_dict.keys()))
            return False
        elif (start_id, end_id) in self.path_dict:
            print("Invalid path: Path {} already exists".format(start_id + '-' + end_id))
            return False
        else:
            status = self.event_dict[start_id].add_reachable(self.event_dict[end_id]) # Checking for requirements
            if status:
                self.path_dict[(start_id, end_id)] = self.embedding_template_dict[embedding_template].copy(embedding_name)
                return True
            else:
                print("Invalid path: Make sure the requirements in event {} is reachable from event {}".format(end_id, start_id))
                return False

    def tune_path_embedding(self, path_id: tuple, prompts: list) -> bool:
        """
        Tune the chosen embedding with extra prompts.
        Params:
            - path_id: Identifier for path, is a tuple of form (start_event, end_event)
            - prompts: List of prompts for tuning
        """
        if self.final:
            print("Sequence Tree was marked as Final. No change can be made to it")
            return False

        if path_id not in self.path_dict:
            print("Path {} does not exists".format(path_id))
            return False
        self.path_dict[path_id].tune_prompts(prompts)
        return True

    def get_template_options(self) -> list:
        """Get the name of all stored templates"""
        return self.embedding_template_dict.keys()

    @staticmethod
    def build_from_json(path_to_json: str, embedding_function: Callable[[list], list]=None) -> 'SequenceTree':
        """
        Rebuild tree from JSON file.
        Params:
            - path_to_json: The string that signifies the path to the jsonified tree
            - embedding_function: Takes input and returns embedding. If not provided, the Tree is marked as Final (cannot be changed)
        """
        tree_dict = {}
        with open(path_to_json, "r") as f:
            tree_dict = json.load(f)

        builder = SequenceTree(tree_dict["name"], tree_dict["dims"], embedding_function)
        for (template_name, template_dict) in tree_dict["embedding_template_dict"].items():
            builder.embedding_template_dict[template_name] = PathEmbedding.from_dict(template_dict, embedding_function)

        for (event_id, event_dict) in tree_dict["event_dict"].items():
            builder.event_dict[event_id] = SequenceEvent.from_dict(event_dict)

        for (path_string, embedding_dict) in tree_dict["path_dict"].items():
            # TODO I fucked up here, json shows "('event0', 'event1')", duct-tape solution
            path_string = path_string[1:-1] # Remove the '(' and the ')'
            path_string = path_string.replace("'", "") # Remove the single quote
            path = tuple(path_string.split(", ")) # Split by ", "
            builder.path_dict[path] = PathEmbedding.from_dict(embedding_dict, embedding_function)
        
        return builder

    def to_dict(self) -> dict:
        """Return the instance in dictionary form to be saved to json"""
        return {
            "name": self.name,
            "embedding_template_dict": {template_id: template.to_dict() for (template_id, template) in self.embedding_template_dict.items()}, # Kinda optional here
            "event_dict": {event_id: event.to_dict() for (event_id, event) in self.event_dict.items()},
            "path_dict": {str(path): embedding.to_dict() for (path, embedding) in self.path_dict.items()},
            "dims": self.dims
        }

    def to_json(self, path_to_json: str) -> None:
        """Saves current Tree configuration to json"""
        with open(path_to_json, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def write_db(self, chroma_client):
        """Write current tree to a ChromaDB collection"""
        collection = chroma_client.get_collection(name=self.name)
        # Only need to add the following nodes to chroma, as the starting state does not need to be defined
        embeddings = []
        documents = []
        metadatas=[]
        ids=[]

        for (path, embedding) in self.path_dict.items():
            target_event_name = path[1]
            target_event = self.event_dict[target_event_name]

            embeddings.append(embedding.get_embedding())
            documents.append(target_event.name)
            ids.append(target_event_name)
            metadatas.append({
                "reachableSequences": json.dumps(target_event.reachable),
                "reaction": target_event.context # TODO Must change later
            })

        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)