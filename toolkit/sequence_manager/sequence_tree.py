"""
Meant to be used alongside chromadb, but any vectordb that uses KNN will work too.
Tunes sequence embeddings to allow the game to more accurately predict player's intentions.
"""
import numpy
from typing import Callable

from state import *

class PathEmbedding:
    """Path embedding dictionary that stores, calculate and allows for retrieval of different preset embeddings"""
    def __init__(self, name: str, embedding_function: Callable[[list], list], dims: int, current_embedding: list=None, current_weight: int=0) -> None:
        """
        Constructor
        Params:
            - name: identifier
            - embedding_function: the function that takes a batch of responses and returns the corresponding batch of embeddings
            - dims: number of dimensionse. e.g. 28 emotions -> 28-d embedding TODO: is this necessary? currently here for consistency
            - current_embedding: the current embedding loaded from json, or pre-determined to bypass tuning. Defaulted to None.
            - current_weight: the current weight loaded from json (use 1 to bypass tuning). If current_weight is 0, the current_embedding will be ignored. Defaulted to 0.
        """
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

    def get_embedding(self) -> list:
        """
        Getter function for embedding attribute. Defaulted to all 0s if no responses have been added.
        """
        return self.embedding

    def add_response(self, responses: list) -> list:
        """
        Add responses to the embedding and calculates new embedding using weighted average. Tunes embedding of SequenceEvent.
        Potentially suffers from density bias. Good response choices will depend on choice of embedding function.
        Params:
            -  responses: the list of responses that should be matched to this path_embedding instance

        Returns: The new embedding
        """
        new_embeddings = self.embed(responses)

        # Update weighted average
        self.embedding = numpy.average([self.embedding] + new_embeddings, 0, [self.weight]+[1]*len(responses)).tolist()
        return self.embedding

    def to_dict(self) -> dict:
        """
        Used to save to json as part of configuration
        Returns: a dict of the form {'name': ..., 'embedding': ..., 'weight': ...} 
        """
        return {
            'name': self.name,
            'embedding': self.embedding,
            'weight': self.weight
        }

class SequenceEvent:
    """
    Describes a game event, or beat that pushes the story forward. The current game state is a sequence_event instance.
    Accessible via the traversal of a sequence_path (unless it is the initial game state).
    Used in-game to determine valid next sequences.
    """
    def __init__(self, id: str, name: str, reaction: str, requirements: dict=None) -> None:
        """
        Constructor.
        Params:
            - id: self-explanatory
            - name: also self-explanatory
            - reaction: the text prompts given as response to player speech. WIP. TODO: May want to be its own Context class
            - requirements: a dictionary containing extra requirement, with key-value pair being the dev-defined id of a quantity and its value
        """
        self.id = id
        self.name = name
        self.context = reaction # Future proofing
        self.requirements = requirements
        # TODO: Process formatted reaction string to allow for a segmented event (Allow player to click on an action button or have another character takeover mid conversation)
    
    def is_valid(self, game_state: State) -> bool:
        """Returns whether this event can be accessed given current game state (can be any child classes of State)"""
        pass

    def to_dict(self) -> dict:
        """
        Used to save to json as part of configuration
        Returns: a dict of the form {'id': ..., 'name': ..., 'context': ...} 
        """
        return {
            'id': self.id, 
            'name': self.name, 
            'context': self.context
        }

class SequenceTree:
    """High-level abstraction of a sequence tree. The game does not see this."""
    def __init__(self, initial_event: SequenceEvent=None) -> None:
        """
        Constructor
        Params:
            - initial_event: the initial event that the player will encounter. Set to None to build tree from json instead. 
        """
        if not initial_event:
            print("No initial event given. Sequence Tree will need to be built from json using .build().")
        self.event = initial_event
        self.paths = {} # Embedding-SequenceEvent key-value pair

    def add_path(self, connecting_sequence: 'SequenceTree', path_embedding: PathEmbedding):
        """Adding sub tree"""
        pass

    def build(path_to_json: str):
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