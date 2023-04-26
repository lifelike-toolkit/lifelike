import chromadb
from chromadb.config import Settings

# Need to reconfigure for deployment
client = chromadb.Client(Settings(chroma_api_impl="rest",
                                  chroma_server_host="localhost",
                                  chroma_server_http_port=8000)) 

collection = client.get_collection(name="test")

query_result = collection.query(
        query_embeddings=[1,1,1,3,4,1,17,36,1,1,2,0,0,1,0,0,0,1,0,0,1,0,3,0,0,0,5,26],
        n_results = 3,
        where={"neutral": True}
    )

print(query_result)