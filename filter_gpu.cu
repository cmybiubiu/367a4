#include "kernels.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <math.h>


__device__ int32_t apply2d_gpu(const int8_t *f, int32_t dimension, const int32_t *original, int32_t width, int32_t height,
                            int row, int column)
{
    // new pixel value
    int32_t pixel = 0;
    // coordinates of the upper left corner
    int32_t upper_left_row = row - dimension/2;
    int32_t upper_left_column = column - dimension/2;
    // multiplying the pixel values with the corresponding values in the Laplacian filter
    for (int r = 0; r < dimension; r ++) { // for each row
        for (int c = 0; c < dimension; c ++) { // for each col
        	int32_t curr_row = upper_left_row + r;
            int32_t curr_col = upper_left_column + c;
            // Pixels on the edges and corners of the image do not have all 8 neighbors. Therefore only the valid
            // neighbors and the corresponding filter weights are factored into computing the new value.
            if (curr_row >= 0 && curr_col >= 0 && curr_row < height && curr_col < width) {
                int coord = curr_row * width + curr_col; // coordinate of the current pixel
                pixel += original[coord] * f[r * dimension + c];
            }
        }
    }
    return pixel;
}


__global__ void reduction(int32_t *smallest, int32_t *biggest, int n) {
  extern __shared__ int32_t sdata[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  if (idx < n) {
    sdata[tid] = smallest[idx];
    sdata[tid + blockDim.x] = biggest[idx];
  }
  __syncthreads();

  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (idx < n && tid % (2*s) == 0) {
      if (tid + s < blockDim.x && idx + s < n) {
        if (sdata[tid] > sdata[tid + s]) {
          sdata[tid] = sdata[tid + s];
        }
        if (sdata[tid + blockDim.x] < sdata[tid + blockDim.x + s]) {
          sdata[tid + blockDim.x] = sdata[tid + blockDim.x + s];
        }
      } 
    }
    __syncthreads();
  }

  if (tid == 0) { // Update golbal variables
    smallest[blockIdx.x] = sdata[0];
    biggest[blockIdx.x] = sdata[blockDim.x];
  }
}


/*------------------------warp reduction kernel 5----------------------------*/
__inline__ __device__
int warp_reduction_max(int val)
{
    for (int offset = warpSize>>1; offset > 0; offset >>= 1)
    {
        val = max(val, __shfl_down(val, offset));
    }
    return val;
}

__inline__ __device__
int warp_reduction_min(int val)
{
    for (int offset = warpSize>>1; offset > 0; offset >>= 1)
    {
        val = min(val, __shfl_down(val, offset));
    }
    return val;
}


__global__ void reduction5(int32_t *max_image, int32_t *min_image, int32_t *biggest, int32_t *smallest, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
  	int tid = threadIdx.x;

  	int max = idx < n? max_image[idx]:-99999;
  	int min = idx < n? min_image[idx]:99999;

  	static __shared__ int shared_max[32]; 
  	static __shared__ int shared_min[32]; 

  	int lane = tid % warpSize;
  	int warp_id = tid / warpSize;

  	max = warp_reduction_max(max); 
  	min = warp_reduction_min(min);

  	if (lane==0) {
        shared_max[warp_id]=max;
        shared_min[warp_id]=min;
  	} 

	__syncthreads(); 

	max = (tid < blockDim.x / warpSize) ? shared_max[lane] : -99999;
    min = (tid < blockDim.x / warpSize) ? shared_min[lane] : 99999;

    if (warp_id==0) {
        max = warp_reduction_max(max); 
        min = warp_reduction_min(min); 
    }


  	if (tid == 0) { 
    	smallest[blockIdx.x] = max;
    	biggest[blockIdx.x] = min;
  	}
}