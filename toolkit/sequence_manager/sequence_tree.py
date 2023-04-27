"""
Meant to be used alongside chromadb, but any vectordb that uses KNN will work too.
Tunes sequence embeddings to allow the game to more accurately predict player's intentions.
"""
import numpy
from typing import Callable


class PathEmbedding:
    """Path embedding dictionary that stores, calculate and allows for retrieval of different preset embeddings"""
    def __init__(self, name: str, embedding_function: Callable[[list], list], dims: int, current_embedding: list=None, current_weight: int=0) -> None:
        """
        Constructor. If loading from dict, use from_dict() instead. 
        Params:
            - name: identifier
            - embedding_function: the function that takes a batch of responses and returns the corresponding batch of embeddings
            - dims: number of dimensionse. e.g. 28 emotions -> 28-d embedding TODO: is this necessary? currently here for consistency
            - current_embedding: the current embedding loaded from json, or pre-determined to bypass tuning. Defaulted to None.
            - current_weight: the current weight loaded from json (use 1 to bypass tuning). If current_weight is 0, the current_embedding will be ignored. Defaulted to 0.
        """
        if not name:
            raise Exception("Path embedding name cannot be None")

        self.name = name

        # Validating embedding function
        test_embeddings = embedding_function(["test string"]) # This must not throw an error
        test_dims = len(test_embeddings[0])
        if test_dims != dims:
            raise Exception("Embedding function is not valid. Returns embedding with {} dims instead of the defined {} dims".format(test_dims, dims))

        self.embed = embedding_function

        if not current_embedding:
            self.embedding = [0]*dims
        else:
            if len(current_embedding) != dims: # Validates embedding
                raise Exception("Embedding is not valid. Embedding has {} dims instead of the defined {} dims".format(test_dims, dims))

            self.embedding = current_embedding

        self.weight = current_weight # The number of responses this embedding represents
        self.n_dims = dims

    @staticmethod
    def from_dict(embedding_dict: dict, embedding_function: Callable[[list], list]) -> 'PathEmbedding':
        """
        Generate up a Path Embedding instance. DO NOT USE TO MAKE DEEP COPY, use copy() instead
        Params:
            - embedding_dict: the result of PathEmbedding.to_dict() instance method, or a dict with the same format
            - embedding_function: the function that takes a batch of responses and returns the corresponding batch of embeddings
        """
        name = embedding_dict["name"]
        embedding = embedding_dict["embedding"]
        dims = len(embedding)
        weight = embedding_dict["weight"]
        return PathEmbedding(name, embedding_function, dims, embedding, weight)

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


class SequenceManager:
    """
    Manages the sequence tree, and interfaces with VectorDB
    """
    def __init__(self) -> None:
        pass


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

    # Import model and tokenizer here
    model = AutoModelForSequenceClassification.from_pretrained('models/bertGoEmotion/') # Assumes the configuration is correct for the use case
    tokenizer = AutoTokenizer.from_pretrained('microsoft/xtremedistil-l6-h384-uncased') # change parameter to your own provided pre-trained model

    sentimentAnalysis = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, top_k=28)

    def embed_responses(responses: list) -> list:
        """
        Simple embedder for testing
        """
        dims = 28
        results = sentimentAnalysis(responses)
        embeddings = []
        for result in results:
            resultEmbedding = [0]*dims
            for labelDict in result:
                emotionIndex = int(labelDict["label"].split("_")[1])
                resultEmbedding[emotionIndex] = labelDict["score"]
            embeddings.append(resultEmbedding)

        assert len(embeddings) == len(results) # Batch size must be the same
        return embeddings

    path_emb = PathEmbedding('test', embed_responses, 28)
    print(path_emb.add_response(["I hate you", "You disgust me"]))
    print(path_emb.to_dict())