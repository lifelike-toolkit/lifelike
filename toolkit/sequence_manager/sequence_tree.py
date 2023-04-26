"""
Meant to be used alongside chromadb, but any vectordb that uses KNN will work too.
Tunes sequence embeddings to allow the game to more accurately predict player's intentions.
"""
import numpy
from typing import Callable

from state import *

from toolkit.database.chromadb_setup import CHROMA_CLIENT


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


class SequenceTree:
    """High-level abstraction of a sequence tree. The game does not need to see this."""
    def __init__(self, id: str=None, name: str=None, reaction: str=None, requirements: dict=None) -> None:
        """
        Constructor. If all constructors are None, you need to build the tree from JSON with SequenceTree.build()
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
        self.paths = {}
        # TODO: Process formatted reaction string to allow for a segmented event (Allow player to click on an action button or have another character takeover mid conversation)

    def build(path_to_json):
        """Rebuild tree using Json to resume progress"""
        pass 

    def add_path(self, connecting_sequence: 'SequenceTree', path_embedding: PathEmbedding) -> None:
        """Adding sub tree. Highly recommend defining all nodes before tying them all together. But a breadth-first approach is possible."""
        embedding = tuple(path_embedding.get_embedding()) # Needs to be tuple to be used as key, convert back in json
        self.paths[embedding] = connecting_sequence # Note that if path_embedding is not unique (extremely rare), this may not give the intended result

    def save(self, path_to_json: str):
        """Saves progress and come back later"""
        pass

    def write_db(self, collection_name: str):
        """Write current tree to a ChromaDB collection"""
        collection = CHROMA_CLIENT.get_collection(name=collection_name)
        # Only need to add the following nodes to chroma, as the starting state does not need to be defined
        embeddings = []
        documents = []
        metadatas=[]
        ids=[]

        for (embedding_tuple, subtree) in self.paths.items():
            embeddings.append(list(embedding_tuple))
            documents.append(subtree.name)
            ids.append(subtree.id)
            metadatas.append({
                "reachableSequences": [seq.id for seq in subtree.paths.values()],
                "reaction": self.context # TODO Must change later
            })

        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)


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