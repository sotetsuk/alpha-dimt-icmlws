#!/bin/sh

cd log
for file in *; do
    echo "${file}"
    cat ${file} | grep BEST | tail -n 1
done
cd ..
