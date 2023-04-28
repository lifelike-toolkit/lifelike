from tree_builder import SequenceTreeBuilder

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

tree_builder = SequenceTreeBuilder("Demo", 28, embed_responses)

# Step 1
print("Define embedding templates and tuning")

embedding_template_name = input("Give a name to this embedding template (q to skip): ")

while embedding_template_name != "q":
    embedding_template = PathEmbedding(embedding_template_name, embed_responses, 28)
    message_list = []
    # Get messages
    message = input("Give a sample response for this embedding template (q to skip): ")
    while message != "q":
        message_list.append(message)
        message = input("Give another sample response for this embedding template (q to skip): ")

    tree_builder.add_embedding_template(embedding_template_name, message_list)

    embedding_template_name = input("Give a name for a new embedding template (q to skip): ")

# Step 2
print("Define sequences")

event_id = input("Give the id for this sequence (use sequence0, sequence1... format) (q to skip): ")
while event_id != "q":
    name = input("Name of sequence: ")
    reaction = input("NPCs' response to this path: ")
    
    tree_builder.add_event(event_id, name, reaction)

    event_id = input("Give the id for another sequence (stay consistent, do not use - in it) (q to skip): ")

print("Add connections")
path_string = input("Give a sequence connection between 2 sequence nodes as id1-id2 (connection is 1 way, id1 to id2) (q to skip): ")

while path_string != 'q':
    embedding_template=input("Choose an available embedding between (watch for typo) {}: ".format(tree_builder.get_template_options()))
    left, right = path_string.split('-')
    tree_builder.add_path(left, right, path_string, embedding_template) # Defaults name to the path_string to ensure uniqueness
    path_string = input("Give another sequence connection between 2 sequence nodes as id1-id2 (connection is 1 way, id1 to id2) (q to skip): ")

# Assuming that sequence0 is the root sequence 
tree_builder.to_json('./test.json')
tree_builder.write_db()