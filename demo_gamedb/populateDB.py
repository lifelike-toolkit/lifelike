import chromadb
from chromadb.config import Settings

# Need to reconfigure for deployment
client = chromadb.Client(Settings(chroma_api_impl="rest",
                                  chroma_server_host="localhost",
                                  chroma_server_http_port=8000)) 

collection = client.create_collection(name="cool_overprotective_dad")

# Mostly here for reference
EMOJIS = [
  'admiration 👏',
  'amusement 😂',
  'anger 😡',
  'annoyance 😒',
  'approval 👍',
  'caring 🤗',
  'confusion 😕',
  'curiosity 🤔',
  'desire 😍',
  'disappointment 😞',
  'disapproval 👎',
  'disgust 🤮',
  'embarrassment 😳',
  'excitement 🤩',
  'fear 😨',
  'gratitude 🙏',
  'grief 😢',
  'joy 😃',
  'love ❤️',
  'nervousness 😬',
  'optimism 🤞',
  'pride 😌',
  'realization 💡',
  'relief😅',
  'remorse 😞', 
  'sadness 😞',
  'surprise 😲',
  'neutral 😐'
];
"""
Metadata will follow the
{"sequenceID": 0, "reachableSequences": [1, 2], "reaction": "Some kinda reaction to getting here"}
SequenceID is the current sequence, reachableSequences is what can be reached from here
"""
collection.add(
    embeddings=[[1.2, 2.3, 4.5], [6.7, 8.2, 9.2]],
    documents=["sequence1", "sequence2", "sequence3", "sequence4", "sequence5"],
    metadatas=[
        {"reachableSequences": [4, 5], "reaction": "Connor: That's my boy! Follow me #Follow Connor to the kitchen# Now what is your opinion on the geopolitical state of kitchen equipments?"}, 
        {"reachableSequences": [], "reaction": "Connor: Damn that's crazy, now get out of my house"}, 
        {"reachableSequences": [], "reaction": "Connor: Seriously, nothing? Guess you aren't the one to ask"},
        {"reachableSequences": [], "reaction": "Connor: Yeah, it's great!"},
        {"reachableSequences": [], "reaction": "Connor: Yeah, bad"}
    ],
    ids=[1, 2, 3, 4, 5]
)