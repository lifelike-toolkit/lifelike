"""
This file contains the interface to manage characters and conversations
"""
import json
import os
import random
from langchain import PromptTemplate, LLMChain


def init(llm) -> None:
    global LLM
    LLM = llm
    
class Characters:
    """
    This class is an interface to manage characters.
    """
    def __init__(self, path: str) -> None:
        """
        @param path: path to the json file
        """
        self.path = path
        if os.path.exists(path):
            self.characters = json.load(open(path, 'r', encoding='utf-8'))
        else:
            self.characters = {}
    
    def is_out(self, name: str) -> ValueError:
        """
        @param name: unique name of the character
        Check if character exists
        """
        if name not in self.characters:
            raise ValueError(f"Character {name} does not exist.")
        
    def is_in(self, name: str) -> ValueError:
        """
        @param name: unique name of the character
        Check if character does not exist
        """
        if name in self.characters:
            raise ValueError(f"Character {name} already exists.")

    def get(self, name: str) -> dict[str, str]:
        """
        @param name: unique name of the character
        Get a characters background from name
        """
        self.is_out(name)
        return self.characters[name]

    def add(self, name: str, background: str) -> None:
        """
        @param name: unique name of the character
        @param background: background of the character
        Add new character to the list
        """
        self.is_in(name)
        self.characters[name] = background

    def update(self, name: str, background: str) -> None:
        """
        @param new_name: name of the character
        @param background: new background of the character
        Revise character
        """
        self.is_out(name)
        self.characters[name] = background

    def delete(self, name) -> None:
        """
        @param name: unique name of the character
        Delete character
        """
        self.is_out(name)
        self.characters.pop(name)

    def __str__(self) -> str:
        """
        Return string representation of the characters
        """
        return str(self.characters)

    def save(self) -> None:
        """
        Save characters to json
        """
        with open(self.path, 'w', encoding='utf-8') as _:
            json.dump(self.characters, _)


class Conversations:
    """
    This class is an interface to manage conversations
    """
    def __init__(self, path: str, characters: Characters) -> None:
        """
        @param path: path to the json file
        """
        self.path = path
        self.valid = characters
        if os.path.exists(path):
            self.conversations = json.load(open(path, 'r', encoding='utf-8'))
        else:
            self.conversations = {}

    def context_out(self, context: str) -> ValueError:
        """
        @param context: unique context of the conversation
        Check if context exists
        """
        if context not in self.conversations:
            raise ValueError(f"Conversation {context} does not exist.")
    
    def context_in(self, context: str) -> ValueError:
        """
        @param context: unique context of the conversation
        Check if context does not exist
        """
        if context in self.conversations:
            raise ValueError(f"Conversation {context} already exists.")
    
    def valid_participants(self, participants: list[str]) -> ValueError:
        """
        @param context: unique context of the conversation
        @param participants: list of character names
        Check if participants are invalid
        """
        for participant in participants:
            if participant not in self.valid.characters:
                raise ValueError(f"Character {participant} does not exist.")

    def get(self, context: str) -> dict[str, list]:
        """
        @param context: unique context of the conversation
        Get participants and log of the conversation
        """
        self.context_out(context)
        return self.conversations[context]

    def new(self, context: str, participants: list[str]) -> None:
        """
        @param context: unique context of the conversation
        @param participants: list of character names
        Create a new conversation
        """
        self.valid_participants(participants)
        self.context_in(context)
        self.conversations[context] = {"participants": participants, "log": []}

    def update(self, context: str, participants: list[str], log: list[(str, str)]) -> None:
        """
        @param context: unique context of the conversation
        @param participants: list of character names
        @param log: list of (speaker, utterance)
        Revise conversation
        """
        self.valid_participants(participants)
        self.context_out(context)
        self.conversations[context] = {"participants": participants, "log": log}

    def delete(self, context: str) -> None:
        """
        @param context: unique context of the conversation
        Delete conversation
        """
        self.context_out(context)
        self.conversations.pop(context)

    def append(self, context: str, speaker: str, utterance: str) -> None:
        """
        @param context: unique context of the conversation
        @param speaker: name of the speaker
        @param utterance: utterance of the speaker
        Append a new utterance to the conversation
        """
        self.valid_participants([speaker])
        self.context_out(context)
        self.conversations[context]["log"].append([speaker, utterance])

    def generate(self, context: str, muted: list[str]) -> str:
        """
        @param context: unique context of the conversation
        @param muted: list of muted characters
        Generate a new utterance
        """
        # TODO: find a smarter way to choose next character
        # TODO: Find a smarter way to get a single response.
        # TODO: memory for conversation log overflow
        # TODO: memory for the context
        # TODO: memory for the character background
        self.valid_participants(muted)
        speaker = random.choice([participant for participant in self.conversations[context]["participants"] if participant not in muted])
        log_str = '\n'.join([f"{speaker}: {utterance}" for speaker, utterance in self.conversations[context]["log"]])

        template = "Current conversation:\n\n" + "{log}" + "\n" + speaker + ":"
        prompt = PromptTemplate(template=template, input_variables=["log"])
        chain = LLMChain(prompt=prompt, llm=LLM)
        output = chain.run(log_str).split('\n')[0].lstrip()
        
        self.append(context, speaker, output)
        return output

    def __str__(self) -> str:
        """
        Return string representation of the conversations
        """
        return str(self.conversations)

    def save(self) -> None:
        """
        Save the file to json
        """
        with open(self.path, 'w', encoding='utf-8') as _:
            json.dump(self.conversations, _)