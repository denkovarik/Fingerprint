from python import Python, PythonObject
from collections import Set
from python import PythonObject
from python.bindings import PythonModuleBuilder
import math
from os import abort


@export
fn PyInit_arch_coder_m() -> PythonObject:
    try:
        var m = PythonModuleBuilder("arch_coder_m")
        m.def_function[reproduce]("reproduce", docstring="Reproduce from dna strands passed in.")
        return m.finalize()
    except e:
        return abort[PythonObject](String("error creating Python Mojo module:", e))


fn reproduce(chosen_filename_py: PythonObject, pop_filename_py: PythonObject, output_filename_py: PythonObject, py_obj_num_offspring: PythonObject, py_obj_num_classes: PythonObject) raises:   
    node_types = init_node_types_m()
    node_ids = init_node_ids_m()
    var chosen_filename: String = String(chosen_filename_py)
    var pop_filename: String = String(pop_filename_py)
    var output_filename: String = String(output_filename_py)
    var empty: String = ""    
    var genotypes: Set[String] = read_file(pop_filename)
    var chosen_genotypes: Set[String] = read_file(chosen_filename)
    var population = { empty }
    var offspring = { empty }
    var offspring_dna = { empty }
    var num_offspring: Int = Int(py_obj_num_offspring)
    var num_output_classes: Int = Int(py_obj_num_classes)
        
    for dna in genotypes:
        var arch = translate_m(dna, num_output_classes, node_types, node_ids)
        population.add(arch)
        
    for dna in chosen_genotypes:
        var arch = translate_m(dna, num_output_classes, node_types, node_ids)
        population.add(arch)
    
    generate_offspring(num_offspring, offspring, offspring_dna, chosen_genotypes, population, 0.05)
    
    write_to_file(offspring_dna, 'offspring_dna.txt')


fn write_to_file(genotypes: Set[String], filename: String) raises:
    with open(filename, 'w') as file:
        for dna in genotypes:
            if dna != "":
                file.write(dna + '\n')


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


def init_node_types_m() -> Dict[String, String]:
    var node_types = Dict[String, String]()
    node_types["10000101"] = "convolution"
    node_types["10011111"] = "linear"
    node_types["00110101"] = "normalize"
    node_types["11101001"] = "activation"
    node_types["01011010"] = "pooling"
    
    return node_types
    

def init_node_ids_m() -> Dict[String, String]:
    node_ids = Dict[String, String]()
    node_ids['input'] = '00000000'
    node_ids['convolution'] = '10000101'
    node_ids['linear'] = '10011111'
    node_ids['normalize'] = '00110101'
    node_ids['activation'] = '11101001'
    node_ids['pooling'] = '01011010'
    node_ids['output'] = '11111111'
    
    return node_ids


fn generate_random_binary_string(length: Int) raises -> String:
    var py = Python.import_module("random")
    
    var binary_string = String("")
    for _ in range(length):
        binary_string += String(py.choice(["0", "1"]))
    
    return binary_string
    
    
fn pow(base: Int, exp: Int) raises -> Int:
    var rslt: Int = 1
    var i: Int = 0
    while i < exp:
        rslt = rslt * base
        i = i + 1
    return rslt


fn read_kernel_size_m(ks: String) raises -> Int:
    var translate_md_ks: Int = 3
    if ks == '000':
        translate_md_ks = 3
    elif ks == '001':
        translate_md_ks = 5
    elif ks == '010':
        translate_md_ks = 7
    elif ks == '011':
        translate_md_ks = 3
    elif ks == '100':
        translate_md_ks = 5
    elif ks == '101':
        translate_md_ks = 7
    elif ks == '110':
        translate_md_ks = 3
    elif ks == '111':
        translate_md_ks = 5
    return translate_md_ks
    

def read_conv_node_m(dna: String, mut found_keys: String, mut i: Int, mut skip: Int, mut input_tensor: PythonObject):
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    
    var in_channels: Int = Int(input_tensor.shape[1])
    var out_chans_exp: Int = Int(dna[i:i+3], 2)
    var out_channels: Int = pow(2, out_chans_exp)
    var ks: Int = read_kernel_size_m(dna[i+3:i+6])
    node = "Conv2D(" + String(in_channels) + "," + String(out_channels) + "," + String(ks) + ")"          
    var valid: Int
    try:
        var m: PythonObject = nn.Conv2d(in_channels, out_channels, ks)
        input_tensor = m(input_tensor)
        valid = 1
    except:
        valid = 0          
    if valid == 1:
        found_keys = found_keys + ", " + node
        skip = 6
        
        
def read_linear_node_m(dna: String, mut found_keys: String, mut i: Int, mut skip: Int, mut input_tensor: PythonObject):
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    var flatener: PythonObject = nn.Flatten()
    
    if len(input_tensor.shape) > 2:
        input_tensor = flatener(input_tensor)
        found_keys = found_keys + ", Flatten()"
    var in_features: Int = Int(input_tensor.shape[1])
    var out_features_exp: Int = Int(dna[i:i+3], 2)
    var out_features: Int = pow(2, out_features_exp)
    var valid: Int
    try:
        var m: PythonObject = nn.Linear(in_features, out_features)
        input_tensor = m(input_tensor)
        valid = 1
    except:
        valid = 0
    if valid == 1:
        skip = 3
        node = "Linear(" + String(in_features) + "," + String(out_features) + ")"
        found_keys = found_keys + ", " + node
        
        
def read_out_linear_node(dna: String, num_class: Int, mut found_keys: String, mut i: Int, mut skip: Int, mut input_tensor: PythonObject):
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    var flatener: PythonObject = nn.Flatten()
    
    if len(input_tensor.shape) > 2:
        input_tensor = flatener(input_tensor)
        found_keys = found_keys + ", Flatten()"
    var in_features: Int = Int(input_tensor.shape[1])
    var out_features: Int = num_class
    var valid: Int
    try:
        var m: PythonObject = nn.Linear(in_features, out_features)
        input_tensor = m(input_tensor)
        valid = 1
    except:
        valid = 0
    if valid == 1:
        skip = 3
        node = "Linear(" + String(in_features) + "," + String(out_features) + ")"
        found_keys = found_keys + ", " + node
        
 
def read_pooling_node_n(dna: String, mut found_keys: String, mut i: Int, mut skip: Int, mut input_tensor: PythonObject):
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    var flatener: PythonObject = nn.Flatten() 

    skip = 1
    var kernelSize: Int = 2
    var stride: Int = 2 
    var valid: Int
    try:
        var max_pool: PythonObject = nn.MaxPool2d(kernelSize, stride)
        input_tensor = max_pool(input_tensor)
        valid = 1
    except:
        valid = 0
    if valid == 1:
        node = "Max_Pooling()"
        found_keys = found_keys + ", " + node
        
     
def find_output_codon_m(dna: String, node_ids: Dict[String, String]) -> Int:
    var stop_ind: Int = -1
    var ind: Int = 8
    while stop_ind == -1 and ind < len(dna):
        if (dna[ind-8:ind] == node_ids['output']):
            stop_ind = ind
        ind += 1
    return stop_ind
    
    
def get_node_type_m(dna: String, ind: Int, stop_ind: Int, node_types: Dict[String, String]) -> String:
    var node: String = ""
    var id: String = dna[ind-8:ind]
    for item in node_types.items():
        if id == item.key:
            node = item.value
    return node
        

def translate_m(dna: String, num_classes: Int, node_types: Dict[String, String], node_ids: Dict[String, String]) -> String:
    torch = Python.import_module("torch")
    nn = Python.import_module("torch.nn")
    
    var found_keys = String("Input")  
    var stop_ind: Int = find_output_codon_m(dna, node_ids)
    var i: Int = 8
    var input_tensor: PythonObject = torch.rand([4, 3, 32, 32])
    var flatener: PythonObject = nn.Flatten()
    
    while i <= stop_ind:
        var skip: Int = 1
        var node: String = get_node_type_m(dna, i, stop_ind, node_types)
                
        if node == 'convolution' and len(input_tensor.shape) == 4 and i + 6 < stop_ind:
            read_conv_node_m(dna, found_keys, i,skip, input_tensor)
        elif node == 'linear':
            read_linear_node_m(dna, found_keys, i,skip, input_tensor)
        elif node == 'normalize' and len(input_tensor.shape) == 4:
            found_keys = found_keys + ", Norm2D()"
        elif node == 'activation':
            found_keys = found_keys + ", ReLU()"
        elif node == 'pooling' and len(input_tensor.shape) == 4:
            read_pooling_node_n(dna, found_keys, i,skip, input_tensor)
        i = i + skip
        
    i = i + 7
    var valid_arch: Int = 0
    while i <= len(dna):
        var skip: Int = 1
        var node: String = get_node_type_m(dna, i, stop_ind, node_types)
        if node == 'linear':
            valid_arch = 1
            read_out_linear_node(dna, num_classes, found_keys, i, skip, input_tensor)
        elif node == 'normalize' and len(input_tensor.shape) == 4:
            found_keys = found_keys + ", Norm2D()"
        elif node == 'activation':
            found_keys = found_keys + ", ReLU()"
        i = i + skip
        
    if valid_arch == 0:
        return ""

    return found_keys
    
    
fn mutate_m(dna: String, mutation_rate: Float32) raises -> String:
    var random = Python.import_module("random")
    var mutant_dna: String = ""
    
    for n in range(len(dna)):
        var nuc: String = dna[n]
        if random.uniform(0,1) < mutation_rate:
            if nuc == "1":
                mutant_dna = mutant_dna + "0"
            else:
                mutant_dna = mutant_dna + "1"
        else: 
            mutant_dna = mutant_dna + nuc
                       
    return mutant_dna
    
    
fn generate_offspring(num_offspring: Int, mut offspring: Set[String], mut offspring_dna: Set[String], mut genotypes: Set[String], population: Set[String], mutation_rate: Float32 = 0.05) raises:    
    node_types = init_node_types_m()
    node_ids = init_node_ids_m()
    
    while len(offspring) < num_offspring:
        for dna in genotypes:
            new_dna = mutate_m(dna + '1', mutation_rate=mutation_rate) 
            var new_phenotype: String = translate_m(new_dna, 10, node_types, node_ids)       
            if new_phenotype not in population:
                offspring.add(new_phenotype)
                offspring_dna.add(new_dna)
            if len(offspring) > num_offspring:
                break
        genotypes = genotypes | offspring_dna
    

