from python import Python, PythonObject
from collections import Set
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
    var torch = Python.import_module("torch")
    
    var input_tensor: PythonObject = torch.rand([4, 3, 32, 32])
    var arc_coder: Arch_Encoder_m = Arch_Encoder_m(input_tensor.shape, num_classes = Int(py_obj_num_classes))   
    var chosen_filename: String = String(chosen_filename_py)
    var pop_filename: String = String(pop_filename_py)
    var output_filename: String = String(output_filename_py)
    var empty: String = ""    
    var genotypes: Set[String] = read_file(pop_filename)
    var chosen_genotypes: Set[String] = read_file(chosen_filename)
    var population = { empty }
    var num_offspring: Int = Int(py_obj_num_offspring)
        
    for dna in genotypes:
        var arch = arc_coder.translate_m(dna)
        population.add(arch)
        
    for dna in chosen_genotypes:
        var arch = arc_coder.translate_m(dna)
        population.add(arch)
    
    var offspring = arc_coder.generate_offspring(num_offspring, chosen_genotypes, population, 0.05)
    
    write_to_file(offspring, 'offspring_dna.txt')


fn write_to_file(offspring: Dict[String, Set[String]], filename: String) raises:
    var offspring_str: String = "{"
    var cnt: Int = 1
    for key in offspring.keys():
        offspring_str = offspring_str + "'" + key + "' : " + offspring[key].__str__()
        if cnt < len(offspring):
            offspring_str = offspring_str + ","
        cnt = cnt + 1
    offspring_str = offspring_str + "}"   
    
    with open(filename, 'w') as file:
        file.write(offspring_str)


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


struct Arch_Encoder_m:
    var node_types: Dict[String, String]
    var node_ids: Dict[String, String]
    var num_output_classes: Int
    var found_keys: String
    var skip: Int
    var input_shape: PythonObject
    var OUTPUT_CODON: String

    fn __init__(out self, input_shape: PythonObject, num_classes: Int) raises:
        self.num_output_classes = num_classes
        self.node_types = Dict[String, String]()
        self.node_ids = Dict[String, String]()
        var torch = Python.import_module("torch")
        self.found_keys = String("Input")
        self.skip = 1
        self.input_shape = input_shape
        self.OUTPUT_CODON = '11111111'
        self.node_types = self.init_node_types_m()
        self.node_ids = self.init_node_ids_m()

    def init_node_types_m(mut self) -> Dict[String, String]:
        var node_types = Dict[String, String]()
        node_types["10000101"] = "convolution"
        node_types["10011111"] = "linear"
        node_types["00110101"] = "normalize"
        node_types["11101001"] = "activation"
        node_types["01011010"] = "pooling"
        return node_types
        
    def init_node_ids_m(mut self) -> Dict[String, String]:
        node_ids = Dict[String, String]()
        node_ids['input'] = '00000000'
        node_ids['convolution'] = '10000101'
        node_ids['linear'] = '10011111'
        node_ids['normalize'] = '00110101'
        node_ids['activation'] = '11101001'
        node_ids['pooling'] = '01011010'
        node_ids['output'] = '11111111'
        return node_ids

    def get_node_type_m(mut self, dna: String, ind: Int, stop_ind: Int) -> String:
        var node: String = ""
        var id: String = dna[ind-8:ind]
        for item in self.node_types.items():
            if id == item.key:
                node = item.value
        return node
        
    def translate_m(mut self, dna: String) -> String:
        torch = Python.import_module("torch")
        nn = Python.import_module("torch.nn")
        
        var input_tensor: PythonObject = torch.rand(self.input_shape)
        self.found_keys = String("Input")  
        var stop_ind: Int = self.find_output_codon_m(dna)
        var i: Int = 8
        var flatener: PythonObject = nn.Flatten()
        
        while i <= stop_ind:
            self.skip = 1
            var node: String = self.get_node_type_m(dna, i, stop_ind)
                    
            if node == 'convolution' and len(input_tensor.shape) == 4 and i + 6 < stop_ind:
                self.read_conv_node_m(dna, i, input_tensor)
            elif node == 'linear':
                self.read_linear_node_m(dna, i, input_tensor)
            elif node == 'normalize' and len(input_tensor.shape) == 4:
                self.found_keys = self.found_keys + ", Norm2D()"
            elif node == 'activation':
                self.found_keys = self.found_keys + ", ReLU()"
            elif node == 'pooling' and len(input_tensor.shape) == 4:
                self.read_pooling_node_n(dna, i, input_tensor)
            i = i + self.skip
            
        i = i + 7
        var valid_arch: Int = 0
        while i <= len(dna):
            self.skip = 1
            var node: String = self.get_node_type_m(dna, i, stop_ind)
            if node == 'linear':
                valid_arch = 1
                self.read_out_linear_node(dna, i, input_tensor)
            elif node == 'normalize' and len(input_tensor.shape) == 4:
                self.found_keys = self.found_keys + ", Norm2D()"
            elif node == 'activation':
                self.found_keys = self.found_keys + ", ReLU()"
            i = i + self.skip
            
        if valid_arch == 0:
            return ""

        return self.found_keys

    fn generate_random_binary_string(mut self, length: Int) raises -> String:
        var py = Python.import_module("random")
        
        var binary_string = String("")
        for _ in range(length):
            binary_string += String(py.choice(["0", "1"]))
        
        return binary_string
           
    fn pow(mut self, base: Int, exp: Int) raises -> Int:
        var rslt: Int = 1
        var i: Int = 0
        while i < exp:
            rslt = rslt * base
            i = i + 1
        return rslt

    fn read_kernel_size_m(mut self, ks: String) raises -> Int:
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
        
    def read_conv_node_m(mut self, dna: String, mut i: Int, mut input_tensor: PythonObject):
        torch = Python.import_module("torch")
        nn = Python.import_module("torch.nn")
        
        var in_channels: Int = Int(input_tensor.shape[1])
        var out_chans_exp: Int = Int(dna[i:i+3], 2)
        var out_channels: Int = self.pow(2, out_chans_exp)
        var ks: Int = self.read_kernel_size_m(dna[i+3:i+6])
        node = "Conv2D(" + String(in_channels) + "," + String(out_channels) + "," + String(ks) + ")"          
        var valid: Int
        try:
            var m: PythonObject = nn.Conv2d(in_channels, out_channels, ks)
            input_tensor = m(input_tensor)
            valid = 1
        except:
            valid = 0          
        if valid == 1:
            self.found_keys = self.found_keys + ", " + node
            self.skip = 6           
            
    def read_linear_node_m(mut self, dna: String, mut i: Int, mut input_tensor: PythonObject):
        torch = Python.import_module("torch")
        nn = Python.import_module("torch.nn")
        var flatener: PythonObject = nn.Flatten()
        
        if len(input_tensor.shape) > 2:
            input_tensor = flatener(input_tensor)
            self.found_keys = self.found_keys + ", Flatten()"
        var in_features: Int = Int(input_tensor.shape[1])
        var out_features_exp: Int = Int(dna[i:i+3], 2)
        var out_features: Int = self.pow(2, out_features_exp)
        var valid: Int
        try:
            var m: PythonObject = nn.Linear(in_features, out_features)
            input_tensor = m(input_tensor)
            valid = 1
        except:
            valid = 0
        if valid == 1:
            self.skip = 3
            node = "Linear(" + String(in_features) + "," + String(out_features) + ")"
            self.found_keys = self.found_keys + ", " + node
            
    def read_out_linear_node(mut self, dna: String, mut i: Int, mut input_tensor: PythonObject):
        torch = Python.import_module("torch")
        nn = Python.import_module("torch.nn")
        var flatener: PythonObject = nn.Flatten()
        
        if len(input_tensor.shape) > 2:
            input_tensor = flatener(input_tensor)
            self.found_keys = self.found_keys + ", Flatten()"
        var in_features: Int = Int(input_tensor.shape[1])
        var valid: Int
        try:
            var m: PythonObject = nn.Linear(in_features, self.num_output_classes)
            input_tensor = m(input_tensor)
            valid = 1
        except:
            valid = 0
        if valid == 1:
            self.skip = 3
            node = "Linear(" + String(in_features) + "," + String(self.num_output_classes) + ")"
            self.found_keys = self.found_keys + ", " + node
            
    def read_pooling_node_n(mut self, dna: String, mut i: Int, mut input_tensor: PythonObject):
        torch = Python.import_module("torch")
        nn = Python.import_module("torch.nn")
        var flatener: PythonObject = nn.Flatten() 

        self.skip = 1
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
            self.found_keys = self.found_keys + ", " + node
                    
    def find_output_codon_m(mut self, dna: String) -> Int:
        var stop_ind: Int = -1
        var ind: Int = 8
        while stop_ind == -1 and ind < len(dna):
            if (dna[ind-8:ind] == self.node_ids['output']):
                stop_ind = ind
            ind += 1
        return stop_ind
              
    fn mutate_m(mut self, dna: String, mutation_rate: Float32) raises -> String:
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
        
    fn generate_offspring(mut self, mut num_offspring: Int, mut genotypes: Set[String], population: Set[String], mutation_rate: Float32) raises -> Dict[String, Set[String]]:    
        var empty: String = ""
        var offspring_keys = { empty }
        num_offspring = num_offspring + 1
        var offspring: Dict[String, Set[String]] = Dict[String, Set[String]]()
        
        while len(offspring_keys) < num_offspring:
            for dna in genotypes:
                new_dna = self.mutate_m(dna + '1', mutation_rate=mutation_rate) 
                var new_phenotype: String = self.translate_m(new_dna)       
                if new_phenotype not in population:
                    if new_phenotype not in offspring_keys:
                        offspring[new_phenotype] = { new_dna }
                        offspring_keys.add(new_phenotype)
                    else:
                        offspring[new_phenotype].add(new_dna)
                if len(offspring_keys) > num_offspring:
                    break
                    
        return offspring

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

