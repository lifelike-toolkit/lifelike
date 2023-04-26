import chromadb
from chromadb.config import Settings

# Need to reconfigure for deployment
client = chromadb.Client(Settings(chroma_api_impl="rest",
                                  chroma_server_host="localhost",
                                  chroma_server_http_port=8000)) 

client.reset()
collection = client.create_collection(name="test")