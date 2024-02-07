#!/bin/bash

counter=0

for file in $(ls -v  iteration-*.png); do
    new_filename="iteration-$counter.png"
    
    mv "$file" "$new_filename"
    echo "Renamed $file to $new_filename"

    ((counter++))
done

