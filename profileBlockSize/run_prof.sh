#!/bin/bash
streams=(1 2 4 8 16)
let "n_streams=${#streams[@]}-1"
for i in `seq 0 $n_streams`; do
  my_stream=${streams[$i]}
  for j in `seq 1 16`; do
    my_thread=$((j * 32))
    echo $my_stream $my_thread
    ./profileBlockSize 10 $my_stream $my_thread >> log.txt
  done
done

grep V1 log.txt | awk '{print $9}'
    
