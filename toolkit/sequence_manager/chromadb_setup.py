"""ChromaDB Client Configuration"""
import chromadb
from chromadb.config import Settings

# Need to reconfigure for deployment
CHROMA_CLIENT = chromadb.Client(Settings(chroma_api_impl="rest",
                                  chroma_server_host="localhost",
                                  chroma_server_http_port=8000)) 

if __name__ == "__main__":
    """Generate necessary collections here"""
    CHROMA_CLIENT.create_collection(name="cool_overprotective_dad") # Make this env