"""
Meant to be used alongside chromadb, but any vectordb that uses KNN will work too.
Tunes sequence embeddings to allow the game to more accurately predict player's intentions.
"""
import numpy
from typing import Callable

class PathEmbedding:
    """An embedding class that can be assigned to sequence paths"""
    def __init__(self, name: str, embedding_function: Callable[[list], list], dims: int, current_embedding: list=None, current_weight: int=0) -> None:
        """
        Constructor
        Params:
            - name: identifier
            - embedding_function: the function that takes a batch of responses and returns the corresponding batch of embeddings
            - dims: number of dimensionse. e.g. 28 emotions -> 28-d embedding TODO: is this necessary? currently here for consistency
            - current_embedding: the current embedding loaded from json. Defaulted to None.
            - current_weight: the current weight loaded from json. If current_weight is 0, the current_embedding will be ignored. Defaulted to 0.
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
        Add responses to the embedding and calculates new embedding using weighted average. Allows for tuning.
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
    High-level abstraction of a sequence event node.
    Describes a game event, or beat that pushes the story forward. The current game state is a sequence_event instance.
    Accessible via the traversal of a sequence_path (unless it is the initial game state).
    """
    def __init__(self) -> None:
        pass

class SequenceTree:
    """High-level abstraction of a sequence tree."""
    def __init__(self, initial_state: SequenceEvent=None) -> None:
        """
        Constructor
        Params:
            - initial_state: the initial event that the player will encounter. Set to None to build tree from json instead. 
        """
        if not initial_state:
            print("No initial state given. Sequence Tree will need to be built from json using .build().")
        self.game_state = initial_state
        self.paths

    def add_path(self, connecting_sequence: 'SequenceTree'):
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