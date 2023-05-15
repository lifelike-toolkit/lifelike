"""
This file contains the interface to manage characters and conversations
"""
import json
import os
import random
from typing import List, Dict, Set
from langchain import PromptTemplate, LLMChain


class Characters:
    """
    This class is an interface to manage characters.
    """
    def __init__(self, path: str) -> None:
        """
        @param path: path to the json file
        @return: None, initializes Characters
        """
        self.path = path
        self.characters = {}
        if os.path.exists(path):
            self.characters = json.load(open(path, 'r', encoding='utf-8'))

    def is_out(self, name: str) -> ValueError:
        """
        @param name: unique name of the character
        @return: ValueError if character does not exist
        """
        if name not in self.characters:
            raise ValueError(f"Character {name} does not exist.")

    def is_in(self, name: str) -> ValueError:
        """
        @param name: unique name of the character
        @return: ValueError if character exists
        """
        if name in self.characters:
            raise ValueError(f"Character {name} already exists.")

    def get(self, name: str) -> str:
        """
        @param name: unique name of the character
        @return: background of the character
        """
        self.is_out(name)
        return self.characters[name]

    def add(self, name: str, background: str) -> None:
        """
        @param name: unique name of the character
        @param background: background of the character
        @return: None, adds character to Characters
        """
        self.is_in(name)
        self.characters[name] = background

    def update(self, name: str, background: str) -> None:
        """
        @param new_name: name of the character
        @param background: new background of the character
        @return: None, updates character in Characters
        """
        self.is_out(name)
        self.characters[name] = background

    def delete(self, name) -> None:
        """
        @param name: unique name of the character
        @return: None, deletes character from Characters
        """
        self.is_out(name)
        self.characters.pop(name)

    def __str__(self) -> str:
        """
        @return: string representation of Characters
        """
        return str(self.characters)

    def save(self) -> None:
        """
        @return: None, saves Characters to json file
        """
        json.dump(self.characters, open(self.path, 'w', encoding='utf-8'))


class Conversations:
    """
    This class is an interface to manage conversations
    """
    def __init__(self, path: str, characters: Characters, llm) -> None:
        """
        @param path: path to the json file
        @param characters: Characters object
        @param llm: langchain llm object
        @return: None, initializes Conversations
        """
        self.path = path
        self.valid = characters
        self.llm = llm
        self.conversations = {}
        if os.path.exists(path):
            self.conversations = json.load(open(path, 'r', encoding='utf-8'))

    def context_out(self, context: str) -> ValueError:
        """
        @param context: unique context of the conversation
        @return: ValueError if context does not exist
        """
        if context not in self.conversations:
            raise ValueError(f"Conversation {context} does not exist.")

    def context_in(self, context: str) -> ValueError:
        """
        @param context: unique context of the conversation
        @return: ValueError if context exists
        """
        if context in self.conversations:
            raise ValueError(f"Conversation {context} already exists.")

    def valid_participants(self, participants: Set[str]) -> ValueError:
        """
        @param context: unique context of the conversation
        @param participants: list of character names
        @return: ValueError if participants are invalid
        """
        invalid = participants - set(self.valid.characters)
        if invalid:
            raise ValueError(f"The participants in {invalid} do not exist.")

    def get(self, context: str) -> Dict[str, any]:
        """
        @param context: unique context of the conversation
        @return: participants and log of the conversation
        """
        self.context_out(context)
        return self.conversations[context]

    def new(self, context: str, participants: Set[str]) -> None:
        """
        @param context: unique context of the conversation
        @param participants: list of character names
        @return: None, creates new conversation
        """
        self.valid_participants(participants)
        self.context_in(context)
        self.conversations[context] = {"participants": participants, "log": []}

    def update(self, context: str, participants: Set[str], log: List[List[str]]) -> None:
        """
        @param context: unique context of the conversation
        @param participants: list of character names
        @param log: list of [speaker, utterance]
        @return: None, updates conversation
        """
        self.valid_participants(participants)
        self.context_out(context)
        self.conversations[context] = {"participants": participants, "log": log}

    def delete(self, context: str) -> None:
        """
        @param context: unique context of the conversation
        @return: None, deletes conversation
        """
        self.context_out(context)
        self.conversations.pop(context)

    def append(self, context: str, speaker: str, utterance: str) -> None:
        """
        @param context: unique context of the conversation
        @param speaker: name of the speaker
        @param utterance: utterance of the speaker
        @return: None, appends utterance to the conversation
        """
        self.valid_participants({speaker})
        self.context_out(context)
        self.conversations[context]["log"].append([speaker, utterance])

    def generate(self, context: str, history: str, muted: Set[str]) -> List[str]:
        """
        @param context: unique context of the conversation
        @param muted: list of muted characters
        @return: speaker and generated utterance
        """
        #TODO: find a smarter way to choose next character
        #TODO: Find a smarter way to get a single response.
        #TODO: memory for conversation log overflow
        #TODO: memory for the context
        #TODO: memory for the character background
        self.valid_participants(muted)

        convo = self.get(context)

        convo_speakers = convo["participants"]
        unmuted = convo_speakers.difference(muted)
        next_speaker = random.sample(unmuted, 1)[0]
        bg = self.valid.get(next_speaker)

        convo_log = convo["log"]
        log_str = '\n'.join([f"{speaker}: {utterance}" for speaker, utterance in convo_log])
        # get last 3 lines of log
        try:
            log_str = '\n'.join(log_str.split('\n')[-3:])
        except IndexError:
            pass

        template = "Context:\n"\
        "{context}\n"\
        "\n"\
        "Background:\n"\
        "{background}\n"\
        "\n"\
        "Relevant pieces of information:\n"\
        "{history}\n"\
        "\n"\
        "(You do not need to use these pieces of information if not relevant)\n"\
        "\n"\
        "Conversation:\n"\
        "{log}\n"\
        "{speaker}:"

        prompt = PromptTemplate(template=template,
                                input_variables=["context", "background",
                                "history", "log", "speaker"])
        chain = LLMChain(prompt=prompt, llm=self.llm)

        output = chain.run({"context": context, "background": bg, "history": history, "log": log_str, "speaker": next_speaker})
        if output != "":
            output = output.split('\n')[0].lstrip()
        self.append(context, next_speaker, output)
        return [next_speaker, output]

    def __str__(self) -> str:
        """
        @return: string representation of Conversations
        """
        return str(self.conversations)

    def save(self) -> None:
        """
        @return: None, saves Conversations to json file
        """
        json.dump(self.conversations, open(self.path, 'w', encoding='utf-8'))
