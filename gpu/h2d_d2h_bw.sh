#!/bin/bash

output_file="testlog.txt"

for ((N=65536; N<=262144; N+=1024))
do
    echo "Running with N=$N"
    ./h2dd2h $N >> "$output_file"
done

