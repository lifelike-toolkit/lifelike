"""
Meant to be used alongside chromadb, but any vectordb that uses KNN will work too.
Tunes sequence embeddings to allow the game to more accurately predict player's intentions.
"""
import numpy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Import model and tokenizer here
model = AutoModelForSequenceClassification.from_pretrained('models/') # Assumes the configuration is correct for the use case
tokenizer = AutoTokenizer.from_pretrained('microsoft/xtremedistil-l6-h384-uncased') # change parameter to your own provided pre-trained model

sentimentAnalysis = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, top_k=28)
print(sentimentAnalysis("I hate you"))