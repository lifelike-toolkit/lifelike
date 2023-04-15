import re

def extract_dialogue(file_path):
    dialogue = []
    with open(file_path, "r") as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check if the line is a character line
        if line.isupper() and not re.search(r'\d', line):
            # Check if the next line is not empty
            if i + 1 < len(lines) and lines[i + 1].strip():
                # Extract the character name and remove content in brackets
                character = re.sub(r'\(.*\)', '', line).strip()
                i += 1
                dialogue_line = ''

                # Collect the dialogue until an empty line is found
                while i < len(lines) and lines[i].strip():
                    dialogue_line += ' ' + lines[i].strip()
                    i += 1

                # Append the dialogue tuple
                dialogue.append((character, dialogue_line.strip()))
            else:
                i += 1
        else:
            i += 1

    return dialogue