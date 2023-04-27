"""
DEPRECATED check tree_builder.py instead
Allows developers to build the sequence tree for the game.
Will be a simple CLI application for now.
TODO: Make this dynamic
"""
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

print("Define embedding classes and tuning")

embedding_classes = {}
embedding_class_name = input("Give a name to this embedding class (q to skip): ")

while embedding_class_name != "q":
    embedding_class = PathEmbedding(embedding_class_name, embed_responses, 28)
    message_list = []
    # Get messages
    message = input("Give a sample response for this embedding class (q to skip): ")
    while message != "q":
        message_list.append(message)
        message = input("Give another sample response for this embedding class (q to skip): ")
    embedding_class.add_response(message_list)
    
    embedding_classes[embedding_class_name] = embedding_class
    embedding_class_name = input("Give a name for a new embedding class (q to skip): ")

print("Define sequences")

sequence_tree_dict = {}
sequence_id = input("Give the id for this sequence (use sequence0, sequence1... format) (q to skip): ")
while sequence_id != "q":
    name = input("Name of sequence: ")
    reaction = input("NPCs' response to this path: ")
    sequence_tree = SequenceTree(sequence_id, name, reaction)
    sequence_tree_dict[sequence_id] = sequence_tree
    sequence_id = input("Give the id for another sequence (stay consistent, do not use - in it) (q to skip): ")

print("Add connections")
sequence_connection = input("Give a sequence connection between 2 sequence nodes as id1-id2 (connection is 1 way, id1 to id2) (q to skip): ")

while sequence_connection != 'q':
    embedding_name=input("Choose an available embedding between (watch for typo) {}: ".format(embedding_classes.keys()))
    embedding_class = embedding_classes[embedding_name]
    left, right = sequence_connection.split('-')
    sequence_left, sequence_right = sequence_tree_dict[left], sequence_tree_dict[right]
    sequence_left.add_path(sequence_right, embedding_class)
    sequence_connection = input("Give another sequence connection between 2 sequence nodes as id1-id2 (connection is 1 way, id1 to id2) (q to skip): ")

# Assuming that sequence0 is the root sequence 
sequence_tree_dict["sequence0"].write_db("test")