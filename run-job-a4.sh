#!/bin/bash

#build the code
make clean
make

#run the job on the GPU server
sbatch job-a4.sh
