"""
Character class
Will assume that the only target of conversation is the player itself
Mostly boilerplate as of rn
"""
import json

class Character:
    def __init__(self, name, personality_dict, context_arr):
        self.name = name
        self.model_id = '' # Talks to openai api, needs to take from config so it's local
        # The rest will be handled by chroma, deal with it then

    def tell(self, dialogue: str) -> str:
        """
        Tells the character something, may trigger change in personality/opinion of player
        Parameters:
            dialogue: str

        Returns: the string representing the response this character has to the dialogue
        """
        response = '' # Handle this with model and chroma
        reaction = {} # Assuming that chroma can deal with it
    
        # stores dialogue in memory as context for later