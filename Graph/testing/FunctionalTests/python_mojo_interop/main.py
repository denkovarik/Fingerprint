import max.mojo.importer
import os
import sys

sys.path.insert(0, "")

import arch_coder_m

pop_filename = 'pop_dna.txt'
chosen_filename = 'chosen_dna.txt'
output_filename = 'offspring_dna.txt'

chosen_dna = {"0001100111000100100111111110100111111111"}
pop_dna = {"0001100111000100100111111110100111111111", "0001100111000100100111111110100111111110"}

chosen_dna_str = ""
pop_dna_str = ""

for dna in pop_dna:
    pop_dna_str = pop_dna_str + dna + ","
with open(pop_filename, 'w') as file:
    file.write(pop_dna_str)
    
for dna in chosen_dna:
    chosen_dna_str = chosen_dna_str + dna + ","
with open(chosen_filename, 'w') as file:
    file.write(chosen_dna_str)


arch_coder_m.reproduce(chosen_filename, pop_filename, output_filename, 10, 10)


with open(output_filename, 'r') as file:
    for line in file:
        # Process the line
        print(line.strip())  # .strip() removes trailing newline characters
