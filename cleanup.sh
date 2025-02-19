#!/bin/bash

TARGET_DIR="/home/skevofilaxc/eqcctpro/mseed/20241215T115800Z_20241215T120100Z"
NUM_TO_KEEP=50

# Count the number of directories
dir_count=$(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)

# If there are more than NUM_TO_KEEP, remove random ones
if [ "$dir_count" -gt "$NUM_TO_KEEP" ]; then
    # List directories, shuffle them, and pick the ones to delete
    dirs_to_delete=$(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d | shuf | head -n $(($dir_count - $NUM_TO_KEEP)))
    
    # Remove selected directories
    echo "Removing the following directories:"
    echo "$dirs_to_delete"
    rm -rf $dirs_to_delete
else
    echo "No directories removed. Only $dir_count directories present."
fi

