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
#define max_threads 1024

void run_kernel2(const int8_t *filter, int32_t dimension, const int32_t *input,
                  int32_t *output, int32_t width, int32_t height)
{
    // Calculate blocks and threads
    int pixel_count = width * height;

    int32_t num_threads = min(max_threads, pixel_count);
    int32_t num_blocks = (pixel_count + num_threads - 1) / num_threads;

    int32_t blocks_reduction = (pixel_count + num_threads - 1) / num_threads;
    // printf("num_threads: %ld, num_blocks: %ld \n", num_threads, num_blocks);

    // kernel 2
    dim3 dimBlock(num_threads, 1, 1);
  	dim3 dimGrid(num_blocks, 1, 1);
  	kernel2<<<dimGrid, dimBlock>>>(filter, dimension, input, output, width, height);

    // init global min & max 
    int32_t *global_min;
    int32_t *global_max;
    cudaMalloc(&global_min, width*height*sizeof(int32_t));
    cudaMalloc(&global_max, width*height*sizeof(int32_t));
    cudaMemcpy(global_min, output, width * height * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(global_max, output, width * height * sizeof(int32_t), cudaMemcpyDeviceToDevice);

    int shMemSize = (num_threads <= 32) ? 4 * num_threads * sizeof(int32_t) : 2* num_threads * sizeof(int32_t);
    reduction<<<blocks_reduction, num_threads, shMemSize>>>(global_min, global_max, pixel_count);

    while (blocks_reduction > 1) {
        int n = blocks_reduction;
        blocks_reduction = (blocks_reduction + max_threads - 1) / max_threads;

        shMemSize = (num_threads <= 32) ? 4 * num_threads * sizeof(int32_t) : 2* num_threads * sizeof(int32_t);
        reduction<<<blocks_reduction, num_threads, shMemSize>>>(global_min, global_max, n);
    }

    // normalize 2
    normalize2<<<num_blocks, num_threads>>>(output, width, height, global_min, global_max);
    cudaFree(global_min);
    cudaFree(global_max);
}


/*row major*/
__global__ void kernel2(const int8_t *filter, int32_t dimension, 
        const int32_t *input, int32_t *output, int32_t width, int32_t height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / width;
	int column = idx % width;

    if (idx < width * height){
        output[idx] = apply2d_gpu(filter, dimension, input, width, height, row, column);
    }
}


__global__ void normalize2(int32_t *image, int32_t width, int32_t height,
        int32_t *smallest, int32_t *biggest)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

  	if (idx < height * width){
    	if (smallest[0] != biggest[0]){
      		image[idx] = ((image[idx] - smallest[0]) * 255) / (biggest[0] - smallest[0]);
    }
  }
}
