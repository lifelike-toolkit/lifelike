import chromadb
from chromadb.config import Settings

# Need to reconfigure for deployment
client = chromadb.Client(Settings(chroma_api_impl="rest",
                                  chroma_server_host="localhost",
                                  chroma_server_http_port=8000)) 

client.reset()
collection = client.create_collection(name="cool_overprotective_dad")

# Mostly here for reference
EMOJIS = [
  'admiration 👏', # 0
  'amusement 😂', # 1
  'anger 😡', # 2
  'annoyance 😒', # 3
  'approval 👍', # 4
  'caring 🤗', # 5
  'confusion 😕', # 6
  'curiosity 🤔', # 7
  'desire 😍',# 8
  'disappointment 😞', #9
  'disapproval 👎', #10
  'disgust 🤮', #11
  'embarrassment 😳', # 12
  'excitement 🤩', #13
  'fear 😨', # 14
  'gratitude 🙏', # 15
  'grief 😢', # 16
  'joy 😃', # 17
  'love ❤️', # 18
  'nervousness 😬', # 19
  'optimism 🤞', # 20
  'pride 😌', # 21
  'realization 💡', # 22
  'relief😅', # 23
  'remorse 😞', # 24
  'sadness 😞', # 25
  'surprise 😲', # 26
  'neutral 😐' # 27
];
"""
Metadata will follow the
{"sequenceID": 0, "reachableSequences": {1: {cool: 2, drunk: 1}, 2: {}}, "reaction": "Some kinda reaction to getting here"}
SequenceID is the current sequence, reachableSequences is what can be reached from this location, as well as extra conditions

Maybe an algorithm to decide how to curb dominance of an emotion for possible answers?
Have users type in possible answers, and the program decides what is the best way to distribute the embeddings target?
"""
collection.add(
    embeddings=[
      [25 if i in [1, 4, 5, 7, 8, 13, 15, 17, 18, 20] else 0 for i in range(28)], # Amusement, Approval, Caring, Curiosity, Desire, Excitement, Gratitude, Joy, Love, Optimism
      [25 if i in [2, 3, 10, 11, 14, 16, 25] else 0 for i in range(28)], # 25 helps create a baseline for answers if neutrality is expected 
      [0]*27+[50], # True neutral, triggers game end. Neutral reaches high very often, but we can raise the target here to curb it
      [25 if i in [0, 4, 8, 13, 15, 17, 18, 20, 21, 23] else 0 for i in range(28)],
      [25 if i in [2, 3, 9, 10, 11, 12, 14, 16, 24, 25, 27 ] else 0 for i in range(28)]
    ],
    documents=["Intro Question", "Early ending", "Neutral ending", "Good Opinion Ending", "Bad opinion ending"],
    metadatas=[
        {"reachableSequences": ["sequence4", "sequence5"], "reaction": "Connor: That's my boy! Follow me #Follow Connor to the kitchen# Now what is your opinion on the geopolitical state of kitchen equipments?"}, 
        {"reachableSequences": [], "reaction": "Connor: Damn that's crazy, now get out of my house"}, 
        {"reachableSequences": [], "reaction": "Connor: Seriously, nothing? Guess you aren't the one to ask"},
        {"reachableSequences": [], "reaction": "Connor: Yeah, it's great!"},
        {"reachableSequences": [], "reaction": "Connor: Yeah, bad"}
    ],
    ids=["sequence1", "sequence2", "sequence3", "sequence4", "sequence5",]
)