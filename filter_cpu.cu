#include "kernels.h"
#include <pthread.h>
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <sched.h>
#define CPU_THREAD 8


/* Helper struct*/
typedef struct filter_t {
    int32_t dimension;
    const int8_t *matrix;
  } filter;

typedef struct common_work_t
{
    const filter *f;
    const int32_t *original_image;
    int32_t *output_image;
    int32_t width;
    int32_t height;
    int32_t max_threads;
    pthread_barrier_t barrier;
} common_work;

typedef struct work_t
{
    common_work *common;
    int32_t id;
} work;


pthread_mutex_t global_min_max_mutex;

int32_t global_min = INT_MAX;
int32_t global_max = INT_MIN;


/* Normalizes a pixel given the smallest and largest integer values
 * in the image */
void normalize_pixel(int32_t *target, int32_t pixel_idx, int32_t smallest,
        int32_t largest)
{
    if (smallest == largest)
    {
        return;
    }

    target[pixel_idx] = ((target[pixel_idx] - smallest) * 255) / (largest - smallest);
}
/*************** COMMON WORK ***********************/
/* Process a single pixel and returns the value of processed pixel
 * TODO: you don't have to implement/use this function, but this is a hint
 * on how to reuse your code.
 * */
int32_t apply2d(const filter *f, const int32_t *original, int32_t *target,
        int32_t width, int32_t height,
        int row, int column)
{
    // new pixel value
    int32_t pixel = 0;
    // coordinates of the upper left corner
    int32_t upper_left_row = row - f->dimension/2;
    int32_t upper_left_column = column - f->dimension/2;
    // multiplying the pixel values with the corresponding values in the Laplacian filter
    for (int r = 0; r < f->dimension; r ++) { // for each row
        for (int c = 0; c < f->dimension; c ++) { // for each col
            int32_t curr_row = upper_left_row + r;
            int32_t curr_col = upper_left_column + c;
            // Pixels on the edges and corners of the image do not have all 8 neighbors. Therefore only the valid
            // neighbors and the corresponding filter weights are factored into computing the new value.
            if (curr_row >= 0 && curr_col >= 0 && curr_row < height && curr_col < width) {
                int coord = curr_row * width + curr_col; // coordinate of the current pixel
                pixel += original[coord] * f->matrix[r * f->dimension + c];
            }
        }
    }
    return pixel;
}


void update_global_min_max(int min, int max) {
    // update global min and global max for normalization
    pthread_mutex_lock(&global_min_max_mutex);
    if (min < global_min) global_min = min;
    if (max > global_max) global_max = max;
    pthread_mutex_unlock(&global_min_max_mutex);
}


void* horizontal_sharding(void *param) {
    work w = *(work*) param;

    // make a copy of the data on local stack
    int height = w.common->height;
    int width = w.common->width;
    int max_threads = w.common->max_threads;
    const filter *f = w.common->f;
    const int32_t *original = w.common->original_image;
    int32_t *target = w.common->output_image;

    // determine start row and end row
    int start_row = w.id * (height / max_threads); // inclusive
    int end_row = (w.id + 1) * (height / max_threads); // exclusive
    if (w.id == max_threads - 1) end_row = height;

    // min and max pixel values for normalization
    int32_t min = INT_MAX;
    int32_t max = INT_MIN;

    // horizontal sharding, row major
    for (int r = start_row; r < end_row; r ++) { // iterate through each row
        for (int c = 0; c < width; c ++) { // iterate through each column
            // process each pixel
            target[r * width + c] = apply2d(f, original, target, width, height, r, c);
            // look for min pixel value
            if (target[r * width + c] < min) min = target[r * width + c];
            // look for max pixel value
            if (target[r * width + c] > max) max = target[r * width + c];
        }
    }


    // update global min and global max for normalization
    update_global_min_max(min, max);

    // wait for all threads to be done with their work
    pthread_barrier_wait(&(w.common->barrier));

    // normalization
    for (int r = start_row; r < end_row; r ++) { // iterate through each row
        for (int c = 0; c < width; c ++) { // iterate through each column
            normalize_pixel(target, r * width + c, global_min, global_max);
        }
    }

    return NULL;
}


/***************** MULTITHREADED ENTRY POINT ******/
/* TODO: this is where you should implement the multithreaded version
 * of the code. Use this function to identify which method is being used
 * and then call some other function that implements it.
 */
void run_best_cpu(const int8_t *filter_data, int32_t dimension, const int32_t *input,
                  int32_t *output, int32_t width, int32_t height)
{
    global_min = INT_MAX;
    global_max = INT_MIN;
    int num_threads = CPU_THREAD;
    // initialize common work
    common_work* cw = (common_work*)malloc(sizeof(common_work));
    const filter f = {dimension, filter_data};
    cw->f = &f;
    cw->original_image = input;
    cw->output_image = output;
    cw->width = width;
    cw->height = height;
    cw->max_threads = num_threads;

    pthread_barrier_init(&(cw->barrier) ,NULL, num_threads);

    // initialize work array
    work** threads_work = (work**)malloc(sizeof(work*) * num_threads);
    for (int i = 0; i < num_threads; i ++) {
        threads_work[i] = (work *) malloc(sizeof(work));
        threads_work[i]->common = cw;
        threads_work[i]->id = i;
    }

    pthread_t threads[num_threads];

    // pin cpu blob 
    int nthreads = get_nprocs();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);


    for (int i = 0; i < num_threads; ++i) {
        pthread_create(&threads[i], NULL, horizontal_sharding, (void *)threads_work[i]);
        CPU_SET(i%nthreads, &cpuset);
        pthread_setaffinity_np(threads[i], sizeof(cpu_set_t), &cpuset);

    }

    // All threads finish their job
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    // clean up
    for (int i = 0; i < num_threads; ++i) {
        free(threads_work[i]);
    }

    free(threads_work);
    free(cw);

    //why delete this?
    //pthread_barrier_destroy(&(cw->barrier));
    
    // restore global max and global min
    global_min = INT_MAX;
    global_max = INT_MIN;
}