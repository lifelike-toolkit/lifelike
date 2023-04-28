import json
from typing import Callable

from .sequence_tree import *

class SequenceTreeBuilder:
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
    def build_from_json(path_to_json: str, embedding_function: Callable[[list], list]=None) -> 'SequenceTreeBuilder':
        """
        Rebuild tree from JSON file.
        Params:
            - path_to_json: The string that signifies the path to the jsonified tree
            - embedding_function: Takes input and returns embedding. If not provided, the Tree is marked as Final (cannot be changed)
        """
        tree_dict = {}
        with open(path_to_json, "r") as f:
            tree_dict = json.load(f)

        builder = SequenceTreeBuilder(tree_dict["name"], tree_dict["dims"], embedding_function)
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


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    from sequence_tree import *

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

    tree_builder = SequenceTreeBuilder.build_from_json("./cod.json", embed_responses, 28)
    tree_builder.to_json('./test_cod.json')
    tree_builder.write_db()