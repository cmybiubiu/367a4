/* ------------
 * This code is provided solely for the personal and private use of 
 * students taking the CSC367 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited. 
 * All forms of distribution of this code, whether as given or with 
 * any changes, are expressly prohibited. 
 * 
 * Authors: Bogdan Simion, Maryam Dehnavi, Felipe de Azevedo Piovezan
 * 
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2020 Bogdan Simion and Maryam Dehnavi
 * -------------
*/

#include "kernels.h"
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <time.h>
#include <stdint.h>
#include <assert.h>

#define max_threads 1024
#define pixels_per_thread 8

#define stream_num 3


void run_kernel5(const int8_t *filter, int32_t dimension, const int32_t *input,
                  int32_t *output, int32_t width, int32_t height)
{   
    // Calculate blocks and threads
    int pixel_count = width * height;
    int32_t num_threads = min(max_threads, pixel_count);
    int32_t num_blocks = ((pixel_count + pixels_per_thread - 1)/pixels_per_thread + (num_threads - 1)) / num_threads;
    int32_t blocks_reduction = (pixel_count + num_threads - 1) / num_threads;

    // kernel 5
    kernel5<<<num_blocks, num_threads>>>(filter, dimension, input, output, width, height);

    int32_t *global_min = NULL;
    int32_t *global_max = NULL;
    cudaMalloc(&global_min, pixel_count*sizeof(int32_t));
    cudaMalloc(&global_max, pixel_count*sizeof(int32_t));
    cudaMemcpy(global_min, output, pixel_count * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(global_max, output, pixel_count * sizeof(int32_t), cudaMemcpyDeviceToDevice);

    reduction5<<< blocks_reduction, num_threads >>>(output, output, global_max, global_min, pixel_count);

    while (blocks_reduction > 1) {
        int n = blocks_reduction;
        blocks_reduction = (blocks_reduction + max_threads - 1) / max_threads;
        reduction5<<< blocks_reduction, num_threads >>>(global_max, global_min, global_max, global_min, n);
    }

    // normalize 5
    normalize5<<<num_blocks, num_threads>>>(output, width, height, global_min, global_max);
    cudaFree(global_min);
    cudaFree(global_max);
}


/* This is your own kernel, you should decide which parameters to add
   here*/
__global__ void kernel5(const int8_t *filter, int32_t dimension, 
        const int32_t *input, int32_t *output, int32_t width, int32_t height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < width * height; i += stride) {
        int row = i / width;
        int column = i % width;
        output[i] = apply2d_gpu(filter,dimension,input,width,height,row,column);
    }
}


__global__ void normalize5(int32_t *image, int32_t width, int32_t height,
        int32_t *smallest, int32_t *biggest)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
  	int stride = blockDim.x * gridDim.x;

  	for (int i = idx; i < width * height; i += stride) {
    	if (smallest[0] != biggest[0]) {
      		image[i] = ((image[i] - smallest[0]) * 255) / (biggest[0] - smallest[0]);
    	}
  	}
}
