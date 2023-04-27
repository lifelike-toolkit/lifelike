from flask import Flask
from flask_cors import CORS
import json

import chromadb
from chromadb.config import Settings

# Need to reconfigure for deployment
client = chromadb.Client(Settings(chroma_api_impl="rest",
                                  chroma_server_host="localhost",
                                  chroma_server_http_port=8000)) 

collection = client.get_collection(name="cool_overprotective_dad")

app = Flask(__name__)
CORS(app)

@app.route('/ping')
def ping():
    # You can just return the string, but I am assuming you'll be returning way more
    return {
        'message': 'ping'
    }

@app.route('/sequence/<embedding>/')
def get_next_sequence(embedding):
    """
    Query the next sequence in the database
    TODO: Some logic for neutral path
    """
    embedding = json.loads(embedding)
    embeddingList = [embedding[str(i)] for i in range(28)] # The incoming JSON is a dict
    print(embeddingList)
    # Query from chromadb
    response = collection.query(
        query_embeddings=embeddingList,
        n_results = 6, # Leave it for now
    )
    # Enable Access-Control-Allow-Origin
    # response.headers.add("Access-Control-Allow-Origin", "*")
    return response