import sys

def count_hyphens(sequence: str) -> int:
    """
    This function takes a string as input and returns the number of hyphens (-) in the sequence.

    :param sequence: The input string to be analyzed.
    :return: The number of hyphens (-) in the sequence.
    """
    return sequence.count('-')


def fill_with_question_marks(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    # Get the length of the reference line (line 2)
    reference_length = len(lines[1].strip())

    # Iterate through even-indexed lines starting from the third line
    for i in range(2, len(lines)):
        line = lines[i].strip()
        if '>' in line:
            continue

        line = line.replace('-','')
        # Calculate the difference between the reference line and the current line
        diff = reference_length - len(line)
        # Add the required number of question marks to fill the gap
        if i % 2 != 0:
            lines[i] = line + '?' * diff + '\n'

    with open(filename, 'w') as file:
        file.writelines(lines)

import ipdb; ipdb.set_trace()
name = sys.argv[1]
fill_with_question_marks(name)