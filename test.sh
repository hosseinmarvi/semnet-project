#!/bin/bash

qsub -t 0-17 -l walltime=24:00:00,mem=32GB -o ./qsub_out/wrapper_evoked_permutation.out -e ./qsub_out/wrapper_evoked_permutation.err -v myvar='/home/sr05/my_semnet/test_function.py'  /home/rf02/rezvan/test1/wrapper_batch.sh
