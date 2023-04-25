"""
Meant to be used alongside chromadb, but any vectordb that uses KNN will work too.
Tunes sequence embeddings to allow the game to more accurately predict player's intentions.
"""
import numpy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Import model and tokenizer here
model = AutoModelForSequenceClassification.from_pretrained('models/bertGoEmotion/') # Assumes the configuration is correct for the use case
tokenizer = AutoTokenizer.from_pretrained('microsoft/xtremedistil-l6-h384-uncased') # change parameter to your own provided pre-trained model

sentimentAnalysis = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, top_k=28)

def labelMapping(label: str) -> str:
    """
    TODO Need to standardize this. 
    For now, alongside a model, one must also provide a custom label Mapping function here
    """
    emotions = [
    'admiration',
    'amusement',
    'anger',
    'annoyance',
    'approval',
    'caring',
    'confusion',
    'curiosity',
    'desire',
    'disappointment',
    'disapproval',
    'disgust',
    'embarrassment',
    'excitement',
    'fear',
    'gratitude',
    'grief',
    'joy',
    'love',
    'nervousness',
    'optimism',
    'pride',
    'realization',
    'relief',
    'remorse',
    'sadness',
    'surprise',
    'neutral']
    pass

def inference2embedding(results: list, dims:int = 28) -> list:
    """
    TODO: Currently assumes the label mapping is LABEL_0 -> admiration and so on
    Prolly a better way to do this, but I'm bruteforcing for now
    Params:
        - result: list containing batch_size * lists of labels and its score
        - dims: number of dimensions. e.g. 28 emotions -> 28-d embedding
    """
    embeddings = []
    for result in results:
        resultEmbedding = [0]*dims
        for labelDict in result:
            emotionIndex = int(labelDict["label"].split("_")[1])
            resultEmbedding[emotionIndex] = labelDict["score"]
        embeddings.append(resultEmbedding)

    assert len(embeddings) == len(results) # Batch size must be the same
    return embeddings

def getMeanEmbedding(responses: list, dims: int = 28) -> list:
    """
    Get the mean embedding of the given responses, the more the better.
    Design to help determine the best embedding for a sequence path.
    Params:
        - responses: the list of string responses given by the dev. Length is arbitrary, but the more the better.
        - dims: number of dimensions. e.g. 28 emotions -> 28-d embedding TODO: make this an environment env
    """
    embeddings = inference2embedding(sentimentAnalysis(responses), dims)
    meanEmbedding = numpy.mean(embeddings, 0).tolist()
    assert len(meanEmbedding) == dims
    return meanEmbedding

class sequence_embedding:
    def __init__(self) -> None:
        self.responses = 

    def 

if __name__ == "__main__":
    print(getMeanEmbedding(["I hate you", "I love you"]))
