import numpy as np
import torch
import torch.nn as nn


INPUT_CODON =  '0000000000000000'
INPUT_SHAPE = [4, 3, 32, 32]
OUTPUT_CONDON = '1111111111111111'
NUM_CLASSES = 10

node_ids = {
    '1000010110011000': 'convolution',
    '1001111100011101': 'linear',
    '0011010110100001': 'normalize',
    '1110100110101010': 'activation',
    #'0101101001101011': 'pooling'
}

print("Node Type       Node ID")
print("--------------------------------")
for id, node_type in node_ids.items():
    print(f"{node_type}: \t{id}")

print("")


node_cnts = {
    'convolution': 0,
    'linear':      0,
    'normalize':   0,
    'activation':  0,
    'pooling':     0
}

def read_kernel_size(ks):
    translated_ks = 3
    if ks == '000':
        translated_ks = 3
    elif ks == '001':
        translated_ks = 5
    elif ks == '010':
        translated_ks = 7
    elif ks == '011':
        translated_ks = 3
    elif ks == '100':
        translated_ks = 5
    elif ks == '101':
        translated_ks = 7
    elif ks == '110':
        translated_ks = 3
    elif ks == '111':
        translated_ks = 5
    return translated_ks

def read_conv_node(dna, s, input_tensor):
    if s + 3 + 3 >= len(dna):
        return None, 1
    
    skip = 0
    node = "convolution("
    in_channels = input_tensor.shape[1]
    node = node + f'in_channels={in_channels}' 
    out_chans_b = dna[s:s+3]
    out_channels_pow = max(2, int(out_chans_b, 2))
    out_channels = pow(2, out_channels_pow)
    skip += 3
    node = node + f',out_channels={out_channels}' 
    kernel_size = read_kernel_size(dna[s+skip:s+skip+3])
    skip += 3
    node = node + f",kernel_size={kernel_size}x{kernel_size}"
    node = node + ")"
    m = nn.Conv2d(in_channels, out_channels, 3)
    output_tensor = m(input_tensor)
    skip += 1
    
    return node, output_tensor, skip

def read_linear_node(dna, s, input_tensor):
    if s >= len(dna):
        return None, input_tensor, 1
    
    skip = 0
    node = "linear("
    in_features = input_tensor.shape[1]
    node = node + f"in_features={in_features}"
    out_features_b = dna[s:s+3]
    out_features_pow = max(1, int(out_features_b, 2))
    out_features = pow(2, out_features_pow)
    m = nn.Linear(in_features, out_features)
    output_tensor = m(input_tensor)
    node = node + f',out_features={out_features}' 
    skip += 3
    node = node + ")"
    skip += 1
    
    return node, output_tensor, skip

def translate_dna(dna: str, node_ids: dict) -> list:
    found_keys = []
    
    input = False
    i = 16
    start_ind = -1
    stop_ind = -1
    
    # Skip to input sequence
    while start_ind == -1 and i < len(dna):
        if dna[i-16:i] == INPUT_CODON:
            start_ind = i
        i += 1
            
    # Find stop codon
    while stop_ind == -1 and i < len(dna):
        if dna[i-16:i] == OUTPUT_CONDON:
            stop_ind = i - 16
        i += 1
        
    if start_ind == -1 or stop_ind == -1:
        return tuple()
    
    input_tensor = torch.rand(INPUT_SHAPE)
    input_node = f'input(input_shape={input_tensor.shape})'
    in_features = 1
    
    found_keys.append(input_node)
    flatener = nn.Flatten()
    
    i = start_ind + 16
    
    while i < stop_ind:
        skip = 1
        if dna[i-16:i] in node_ids.keys():
            id = dna[i-16:i]
            node = node_ids[id]        
            if node == 'convolution' and len(input_tensor.shape) == 4:
                node, out_tensor, skip = read_conv_node(dna, i, input_tensor)
                found_keys.append(node)
                input_tensor = out_tensor
            elif node == 'linear':
                if len(input_tensor.shape) > 2:
                    input_tensor = flatener(input_tensor)
                    found_keys.append('flatten')
                node, out_tensor, skip = read_linear_node(dna, i, input_tensor)
                found_keys.append(node)
                input_tensor = out_tensor
            elif node == 'normalize':
                found_keys.append('batch_norm')
            elif node == 'activation':
                found_keys.append('relu activation')
        i += skip
    
    if len(input_tensor.shape) > 2:
        flatener = nn.Flatten()
        input_tensor = flatener(input_tensor)
        found_keys.append('flatten')
    
    in_features = input_tensor.shape[1]
    
    found_keys.append(f'linear(in_features={in_features}, out_features={NUM_CLASSES}')
    found_keys.append(f'output')
    
    # Return only the keys in order as a tuple
    return tuple(key for key in found_keys)


rnas = set()

for i in range(1000):
    dna_array = np.random.randint(0, 2, size=2048)
    dna = INPUT_CODON + ''.join(dna_array.astype(str)) + OUTPUT_CONDON

    for id, node_type in node_ids.items():
        node_cnts[node_type] += dna.count(id)
        
    rnas.add(translate_dna(dna, node_ids))

print("\n")
print("Gene Pool")
print("-----------------------------------")
for rna in rnas:
    print(rna)