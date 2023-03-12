# File: reformatSynFingerprintDataset.py
# Description: 
# Reformats the Synthetic Fingerprint Dataset into a more intuitive format. 
# When this dataset was created by the Anguli software, each unique fingerprint 
# is named by a given number. The impressions of each unique fingerprint for 
# the entire dataset are put into separate folders, so each unique fingerprint 
# is spread out over multiple folders instead of them all being located in the 
# same folder. This script will reformat this dataset so that all impressions 
# of each unique fingerprint are located in the same folder, that way the folder 
# itself is used to identify each unique fingerprint.
# Usage:
#   python reformatSynFingerprintDataset.py inDirPath outDirPath


import sys, os
import pathlib
import shutil
from progress.bar import Bar
from progress.spinner import Spinner

def printUsage():
	"""
	Prints the usage stagement for the script.
	"""
	usage = "python reformatSynFingerprintDataset.py $inDirPath $outDirPath"
	print(usage)


# Validate command line arguments
if len(sys.argv) != 3:
	print("Invalid number command line arguments.")
	print("Usage:\n\t", end="")
	printUsage()
	exit()

if not os.path.isdir(sys.argv[1]):
	print(sys.argv[1] + " does not exist")
	exit()

if sys.argv[1] == sys.argv[2]:
	print("Input and output directory paths cannot be the same") 


inDir = sys.argv[1]
outDir = sys.argv[2]

files2Copy = []

spinner = Spinner("Discovering Files to Copy: ")

# Iterate over input dir to get file src and dest info
for d1 in os.listdir(inDir):
	d1Path = os.path.join(inDir, d1)
	for d2 in os.listdir(d1Path):
		d2Path = os.path.join(d1Path, d2)
		for d3 in os.listdir(d2Path):
			d3Path = os.path.join(d2Path, d3)
			for f in os.listdir(d3Path):
				fId = f.split(".")[0]
				fileName = "Fingerprint_" + fId + "_" + d2 + "." + f.split(".")[1]
				inPath = os.path.join(d3Path, f)
				outDirPath = os.path.join(outDir, d1, fId)
				outPath = os.path.join(outDirPath, fileName)
				files2Copy.append({"src": inPath, "dest": outPath, "destDir": outDirPath})
		spinner.next()


# Copy the files
with Bar("Copying Files: ", max=len(files2Copy)) as bar:
	for i in range(len(files2Copy)):
		if not os.path.isdir(files2Copy[i]["destDir"]):
			pathlib.Path(files2Copy[i]["destDir"]).mkdir(parents=True, exist_ok=True)
		shutil.copyfile(files2Copy[i]["src"], files2Copy[i]["dest"])
		bar.next()


