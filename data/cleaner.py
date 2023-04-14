import re

# open and read file
def extract_dialogue(filename):
    with open(filename, 'r') as f:
        script = f.read()
    
    # Here is the structure:
    # 1. The character name is always in all caps
    # 2. The character name has a new line before it
    # 3. The character name may or may not contain a space, a . a ( a ) or a '
    # 4. The character name is always followed by a new line
    # 5. The dialogue of the character follows on the next line of the character name
    # 6. The dialogue continues until there is an empty line
    # 7. remove all text contained in parentheses

    # write a regex to match the above

    dialogues = re.findall(r'([A-Z\s\.\(\)\'\,]+)\n([^\n]+)', script)

    # remove all text contained in parentheses
    dialogues = [(character, re.sub(r'\([^)]*\)', '', dialogue)) for character, dialogue in dialogues]

    # remove all whitespace and new lines
    dialogues = [(character, dialogue.strip()) for character, dialogue in dialogues]

    return dialogues

dialogues = extract_dialogue('Zootopia.txt')

for character, dialogue in dialogues:
    print('[' + character + ']')
    print(dialogue)