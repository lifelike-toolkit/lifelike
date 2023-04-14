"""Controllers for character class"""
from character import Character
from utils import getCharacter

def newCharacter(name, personality_dict: dict, context_arr: list) -> Character:
    return Character(name, personality_dict, context_arr)

def tellCharacter(name, dialogue) -> str:
    character = getCharacter(name)
    return character.tell(dialogue)