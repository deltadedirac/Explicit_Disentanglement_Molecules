import sys

'''
this is the code to convert into one line 


------ THIS CHUNK IS TO PASS THE RAW SEQUENCES IN ONE LINE QUITING GAPS---------------

#!/bin/awk -f

awk '/^>/ { 
    if (seq) {
        print seq
        seq=""
    }
    printf("%s\t", $0)
    next
}
{
    gsub("-", "", $0)
    gsub(".", "", $0)
    seq = seq $0
}
END {
    if (seq) {
        print seq
    }
}' your_input.fasta




------ THIS CHUNK IS TO PASS THE ALIGNMENT'S GAPS IN ONE LINE ---------------
awk '/^>/ { 
    if (seq) {
        print seq
        seq=""
    }
    printf("%s\t", $0)
    next
}
{
    seq = seq $0
}
END {
    if (seq) {
        print seq
    }
}' your_input.fasta


'''

def fill_with_question_marks(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    # Get the length of the reference line (line 2)
    reference_length = len(lines[1].strip())

    # Iterate through even-indexed lines starting from the third line
    for i in range(2, len(lines)):
        line = lines[i].strip()
        # Calculate the difference between the reference line and the current line
        diff = reference_length - len(line)
        # Add the required number of question marks to fill the gap
        if i % 2 != 0:
            lines[i] = line + '?' * diff + '\n'

    with open(filename, 'w') as file:
        file.writelines(lines)

# Example usage:
name = sys.argv[1]
fill_with_question_marks(name)
