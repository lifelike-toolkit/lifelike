# lifelike
A toolkit that allows for the creation of "lifelike" characters that you can interact with and change how they behave towards you


## Brain Useage
The `Characters` class requires a file path to a JSON file where the characters data will be saved. To create a `Characters` object, initialize it as follows:
```python
import brain
characters = brain.Characters("/path/to/your/characters.json")
```
Once the object is initialized, the following methods can be used:
- `add(self, name: str, background: str) -> None: ` adds a character to the character class. Needs a unique name and background written in natural language.
- `get(self, name: str) -> str: ` returns the background of a character.
- `update(self, name: str, background: str) -> None: ` updates the background of a character.
- `delete(self, name) -> None: ` removes a character from the Characters class
- `save(self) -> None: ` saves all characters to path specified in initiailization

The `Conversations` class requires a file path to a JSON file where the conversations data will be saved, a `Characters` object ss well as a `langchain.llms` object. Initiailize it as follows:
```python
# using 'characters' and 'import brain' from above.
from langchain.llms import LlamaCpp
llm = LlamaCpp(model_path='ggml-model-q4_0.bin') # Can be any LLM

conversations = brain.Conversations("/path/to/your/conversations.json",
                                    characters,
                                    llm)
```
Once the object is initialized, the following methods can be used:
- `new(self, context: str, participants: Set[str]) -> None: ` with a unique context and participants creates a new conversation in Conversations.
- `get(self, context: str) -> Dict[str, any]:` returns the participants and log of a conversation in the structure of `{"participants":set, "log":[]}` from a conversation with context
- `update(self, context: str, participants: Set[str], log: List[List[str]]) -> None: ` with a unique context update the participants (set) and log (list of [character, utterance] in sequential order)
- `delete(self, context: str) -> None: ` deletes a conversation from Conversations
- `append(self, context: str, speaker: str, utterance: str) -> None:` add utterance from speaker in conversation with unique context
- `generate(self, context: str, muted: Set[str]) -> List[str]:` given a conversations context and muted characters have an unmuted character say something and add it to the conversation. returns `[speaker, utterance]`
- `save(self) -> None: ` saves all conversations to a path specialized in initialization
