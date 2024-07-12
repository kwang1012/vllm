#!/bin/bash

batchSizes=(16 32 64 128 256)
maxTokens=(100 200 300 400 500 500 1000)

for i in $(seq 2 2);
do
    for j in $(seq 3 4);
    do
        if [ $i -eq 1 ]
        then
            tp=$j
            pp=1
        else
            tp=1
            pp=$j
        fi
        if [ $tp -eq 3 ] || [ $pp -eq 3 ] # not divisible by 3
        then
            continue
        fi
        for batchSize in "${batchSizes[@]}"
        do
            for maxToken in "${maxTokens[@]}"
            do
                echo Parsing experiment log, batch: $batchSize, tp: $tp, pp: $pp, maxTokens: $maxToken
                python log2csv.py results/profiling/result-$tp-$pp-$batchSize-$maxToken
            done
        done
        for batchSize in "${batchSizes[@]}"
        do
            for maxToken in "${maxTokens[@]}"
            do
                echo Plotting experiment figure, batch: $batchSize, tp: $tp, pp: $pp, maxTokens: $maxToken
                python plot_csv.py results/profiling/result-$tp-$pp-$batchSize-$maxToken &
            done
        done
    done
done