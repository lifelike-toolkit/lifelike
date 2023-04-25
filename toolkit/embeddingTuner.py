"""
Meant to be used alongside chromadb, but any vectordb that uses KNN will work too.
Tunes sequence embeddings to allow the game to more accurately predict player's intentions.
"""
import numpy
import torch
from transformers import AutoTokenizer, AutoModel

# Import model and tokenizer here
model = AutoModel.from_pretrained('../models/bertGoEmotion') # Assumes the configuration is correct for the use case
tokenizer = AutoTokenizer.from_pretrained('microsoft/xtremedistil-l6-h384-uncased') # change parameter to your own provided pre-trained model

