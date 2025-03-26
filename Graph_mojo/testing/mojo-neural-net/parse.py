import csv
import numpy as np

def normalize(arr):
    arr = arr.astype(float)
    arr_min = np.amin(arr)
    arr_max = np.amax(arr)
    arr_norm = (arr - arr_min) / (arr_max - arr_min) 
    return arr_norm  

data = []

#Load in data and normalize
with open('letter-data.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')     
    for row in readCSV:
        row[0] = ord(row[0]) - 65
        row = np.asarray(row)
        row = row.astype(float)
        data.append(row)                     
    data = np.asmatrix(data)
    for col in range(1,17):
        data[:,col] = normalize(data[:,col])
print(data[0][0].shape, data.shape)
print('done parsing')

np.savetxt("letters-data-normalized.txt",data)
np.savetxt("letters-inputs.txt", data[:,1:17])
np.savetxt("letters-validate.txt", data[:,0])