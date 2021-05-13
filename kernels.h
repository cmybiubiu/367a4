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

#ifndef __KERNELS__H
#define __KERNELS__H

/* TODO: you may want to change the signature of some or all of those functions,
 * depending on your strategy to compute min/max elements.
 * Be careful: "min" and "max" are names of CUDA library functions unfortunately,
 * so don't use those for variable names.*/
void run_best_cpu(const int8_t *filter, int32_t dimension, const int32_t *input,
                  int32_t *output, int32_t width, int32_t height);

__device__ int32_t apply2d_gpu(const int8_t *f, int32_t dimension, const int32_t *original, int32_t width, int32_t height,
                            int row, int column);
__global__ void reduction(int32_t *smallest, int32_t *biggest, int n);

void run_kernel1(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height);
__global__ void kernel1(const int8_t *filter, int32_t dimension, 
        const int32_t *input, int32_t *output, int32_t width, int32_t height);
__global__ void normalize1(int32_t *image, int32_t width, int32_t height,
        int32_t *smallest, int32_t *biggest);

void run_kernel2(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height);
__global__ void kernel2(const int8_t *filter, int32_t dimension, 
        const int32_t *input, int32_t *output, int32_t width, int32_t height);
__global__ void normalize2(int32_t *image, int32_t width, int32_t height,
        int32_t *smallest, int32_t *biggest);

void run_kernel3(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height);
__global__ void kernel3(const int8_t *filter, int32_t dimension, 
        const int32_t *input, int32_t *output, int32_t width, int32_t height);
__global__ void normalize3(int32_t *image, int32_t width, int32_t height,
        int32_t *smallest, int32_t *biggest);

void run_kernel4(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height);
__global__ void kernel4(const int8_t *filter, int32_t dimension, 
        const int32_t *input, int32_t *output, int32_t width, int32_t height);
__global__ void normalize4(int32_t *image, int32_t width, int32_t height,
        int32_t *smallest, int32_t *biggest);


/* This is your own kernel, you should decide which parameters to add
   here*/
void run_kernel5(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height);

__global__ void kernel5(const int8_t *filter, int32_t dimension, 
        const int32_t *input, int32_t *output, int32_t width, int32_t height);

__global__ void normalize5(int32_t *image, int32_t width, int32_t height,
        int32_t *smallest, int32_t *biggest);

__global__ void reduction5(int32_t *max_image, int32_t *min_image, int32_t *biggest, int32_t *smallest, int n);




#endif
