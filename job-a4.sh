#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --partition=csc367-compute
#SBATCH --job-name a4 
#SBATCH --output=a4_%j.out

./solution.out -i Images/4mb_image.pgm -o 4mb_output.pgm
./solution.out -i Images/big_sample.pgm -o big_output.pgm
