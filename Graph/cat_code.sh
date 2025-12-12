#!/bin/bash

# Check if a directory path is provided as an argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

# Store the directory path from command line argument
dir="$1"

echo $dir

# Check if the provided path is a valid directory
if [ ! -d "$dir" ]; then
    echo "Error: '$dir' is not a valid directory"
    exit 1
fi

# List of files to cat in the specified directory
files_to_cat=(
    "GraphBuilder.jsx"
    "GraphCanvas.jsx"
    "GraphEditor.jsx"
    "NodeList.jsx"
    # Add more filenames as needed
)

echo "Processing directory: $dir"
echo "Listing contents of $dir:"
ls "$dir"
echo "----------------------------------------"

# Loop through each file in the list
for file in "${files_to_cat[@]}"; do
    file_path="$dir/$file"
    if [ -f "$file_path" ]; then
	echo ""
        echo "Contents of $file:"
	echo "----------------------------------------"
        cat "$file_path"
        echo "----------------------------------------"
    else
        echo "File $file not found in $dir"
    fi
done

echo "Finished processing $dir"
echo "========================================="
