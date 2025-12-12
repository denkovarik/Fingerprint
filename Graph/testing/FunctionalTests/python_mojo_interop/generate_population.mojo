from arch_coder_m import Arch_Encoder_m
from python import Python, PythonObject
from collections import Set


fn write_to_file(pop_dna: Set[String], filename: String) raises:
    with open(filename, 'a') as file:
        for dna in pop_dna:
            if dna != "":
                file.write(dna + ",")


fn read_file(filename: String) raises -> Set[String]:
    var empty: String = ""
    var genotypes: Set[String] = { empty } 
    
    with open(filename, "r") as f:
        contents = f.read()
        var dna: String = ""
        for i in range(len(contents)):
            c = contents[i]
            if c == ",":
                genotypes.add(dna)
                dna = ""
            elif c == '1' or c == '0':
                dna = dna + c
                
    return genotypes
    

def main():    
    var torch = Python.import_module("torch")
    
    var input_tensor: PythonObject = torch.rand([4, 3, 32, 32])
    var arch_encoder: Arch_Encoder_m = Arch_Encoder_m(input_tensor.shape, num_classes = 10)
    var empty: String = ""    
    var population = { empty }
    var out_filename: String = 'graph_dna.txt'
    var max_dna_len: Int = 80
    
    
    print("Node Type\tNode ID")
    print("--------------------------------")
    for id in arch_encoder.node_types:
        var node_type: String = arch_encoder.node_types[id]
        print(id + "\t" + node_type)
        
        
    # Seeding the population
    var pop_genotypes_all = read_file(out_filename)
    var pop_genotypes: Set[String] = { empty } 
    var dna: String = ''
    #dna = dna + arch_encoder.OUTPUT_CODON + String(arch_encoder.node_ids['linear']) + String(arch_encoder.node_ids['activation']) + '11111111' + '11111111'
    dna = dna + '00110101' + '10000101' + '010000' + '11101001' + '01011010' + arch_encoder.OUTPUT_CODON + String(arch_encoder.node_ids['linear']) + String(arch_encoder.node_ids['activation']) 
    
    if len(dna) > max_dna_len:
        max_dna_len = len(dna)
    
    pop_genotypes_all.add(dna)
    
    for dna in pop_genotypes_all:
        if len(dna) <= max_dna_len and len(dna) > 0:
            pop_genotypes.add(dna)
            var phenotype = arch_encoder.translate_m(dna)
            population.add(phenotype)


    print("\n")
    print("Generation 1")
    print("-----------------------------------")
    for phenotype in population:
        print(phenotype)
        
        
    var num_generations: Int = 5000
    var generation_num: Int = 2
    var num_offspring: Int = 10

    with open(out_filename, 'w') as file:
        for dna in pop_genotypes_all:
            file.write(dna + ",")
    
        while generation_num <= num_generations: 
            var offspring = arch_encoder.generate_offspring(num_offspring, pop_genotypes, population, 0.05)

            for phenotype in offspring:
                var valid: Int = 0
                for dna in offspring[phenotype]:
                    if len(dna) <= max_dna_len and dna not in pop_genotypes:
                        pop_genotypes.add(dna)
                        file.write(dna + ",")
                        valid = 1
                if valid == 1:
                    population.add(phenotype)
                
            #print("Generation " + String(generation_num) + "\n\tNumber of phenotypes = " + String(len(population)))
            print("\n\n")
            print("Generation " + String(generation_num))
            print("-----------------------------------")
            for child in population:
                if child != "":
                    print(child)
        
            generation_num = generation_num + 1
