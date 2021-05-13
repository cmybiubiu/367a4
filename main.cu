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

#include <stdio.h>
#include <string>
#include <unistd.h>

#include "pgm.h"
#include "clock.h"
#include "kernels.h"


/* laplacian */
int8_t lp3_m[] =
    {
        0, 1, 0,
        1, -4, 1,
        0, 1, 0,
    };

int8_t lp5_m[] =
    {
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, 24, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
    };

/* Laplacian of gaussian */
int8_t log_m[] =
    {
        0, 1, 1, 2, 2, 2, 1, 1, 0,
        1, 2, 4, 5, 5, 5, 4, 2, 1,
        1, 4, 5, 3, 0, 3, 5, 4, 1,
        2, 5, 3, -12, -24, -12, 3, 5, 2,
        2, 5, 0, -24, -40, -24, 0, 5, 2,
        2, 5, 3, -12, -24, -12, 3, 5, 2,
        1, 4, 5, 3, 0, 3, 5, 4, 1,
        1, 2, 4, 5, 5, 5, 4, 2, 1,
        0, 1, 1, 2, 2, 2, 1, 1, 0,
    };
/* Dimension */
int32_t dimension = 5;

/* Use this function to print the time of each of your kernels.
 * The parameter names are intuitive, but don't hesitate to ask
 * for clarifications.
 * DO NOT modify this function.*/
void print_run(float time_cpu, int kernel, float time_gpu_computation,
        float time_gpu_transfer_in, float time_gpu_transfer_out)
{
    printf("%12.6f ", time_cpu);
    printf("%5d ", kernel);
    printf("%12.6f ", time_gpu_computation);
    printf("%14.6f ", time_gpu_transfer_in);
    printf("%15.6f ", time_gpu_transfer_out);
    printf("%13.2f ", time_cpu/time_gpu_computation);
    printf("%7.2f\n", time_cpu/
            (time_gpu_computation  + time_gpu_transfer_in + time_gpu_transfer_out));
}

int main(int argc, char **argv)
{
    int c;
    std::string input_filename, cpu_output_filename, base_gpu_output_filename;
    if (argc < 3)
    {
        printf("Wrong usage. Expected -i <input_file> -o <output_file>\n");
        return 0;
    }

    while ((c = getopt (argc, argv, "i:o:")) != -1)
    {
        switch (c)
        {
            case 'i':
                input_filename = std::string(optarg);
                break;
            case 'o':
                cpu_output_filename = std::string(optarg);
                base_gpu_output_filename = std::string(optarg);
                break;
            default:
                return 0;
        }
    }

    pgm_image source_img;
    init_pgm_image(&source_img);

    if (load_pgm_from_file(input_filename.c_str(), &source_img) != NO_ERR)
    {
       printf("Error loading source image.\n");
       return 0;
    }

    /* Do not modify this printf */
    printf("CPU_time(ms) Kernel GPU_time(ms) TransferIn(ms) TransferOut(ms) "
            "Speedup_noTrf Speedup\n");

    float time_cpu;
    {
        std::string cpu_file = cpu_output_filename;
        pgm_image cpu_output_img;
        copy_pgm_image_size(&source_img, &cpu_output_img);

        // Start time
        struct timespec start, stop;
        clock_gettime(CLOCK_MONOTONIC, &start);
        run_best_cpu(lp5_m, dimension, source_img.matrix, cpu_output_img.matrix, source_img.width, source_img.height);  // From kernels.h
        // End time
        clock_gettime(CLOCK_MONOTONIC, &stop);
        time_cpu = ((stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / 1000000000) * 1000;
        // print_run(time_cpu, 0, 0, 0, 0);
        save_pgm_to_file(cpu_file.c_str(), &cpu_output_img);
        destroy_pgm_image(&cpu_output_img);
    }

    /*-------------------- Kernel 1 ----------------------*/
    {
        std::string gpu_file = "1" + base_gpu_output_filename;
        pgm_image gpu_output_img;
        copy_pgm_image_size(&source_img, &gpu_output_img);
        int32_t *device_input = NULL;
        int32_t *device_output = NULL;
        int8_t *device_filter = NULL;
        size_t size = gpu_output_img.width * gpu_output_img.height * sizeof(int32_t);
        cudaMalloc((void **)&device_filter, dimension*dimension*sizeof(int8_t));
        cudaMalloc((void **)&device_input, size);
        cudaMalloc((void **)&device_output, size);
        float transfer_in, transfer_out, computation;
 
        Clock clock;
        clock.start();
        cudaMemcpy(device_filter, lp5_m, dimension*dimension*sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(device_input, source_img.matrix, size, cudaMemcpyHostToDevice);
        cudaMemcpy(device_output, gpu_output_img.matrix, size, cudaMemcpyHostToDevice);
        transfer_in = clock.stop();
 
        // Start time
        clock.start();
        run_kernel1(device_filter, dimension, device_input, device_output, gpu_output_img.width, gpu_output_img.height);
        // End time
        computation = clock.stop();
    
        clock.start();
        cudaMemcpy(gpu_output_img.matrix, device_output, size, cudaMemcpyDeviceToHost);
        transfer_out = clock.stop();
 
        print_run(time_cpu, 1, computation*1000, transfer_in*1000, transfer_out*1000); 
        save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
        cudaFree(device_filter);
        cudaFree(device_input);
        cudaFree(device_output);
        destroy_pgm_image(&gpu_output_img);
    }

    /*-------------------- Kernel 2 ----------------------*/
    {
        std::string gpu_file2 = "2" + base_gpu_output_filename;
        pgm_image gpu_output_img;
        copy_pgm_image_size(&source_img, &gpu_output_img);
        int32_t *device_input = NULL;
        int32_t *device_output = NULL;
        int8_t *device_filter = NULL;
        size_t size = gpu_output_img.width * gpu_output_img.height * sizeof(int32_t);
        cudaMalloc((void **)&device_filter, dimension*dimension*sizeof(int8_t));
        cudaMalloc((void **)&device_input, size);
        cudaMalloc((void **)&device_output, size);
        float transfer_in, transfer_out, computation;
 
        Clock clock;
        clock.start();
        cudaMemcpy(device_filter, lp5_m, dimension*dimension*sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(device_input, source_img.matrix, size, cudaMemcpyHostToDevice);
        cudaMemcpy(device_output, gpu_output_img.matrix, size, cudaMemcpyHostToDevice);
        transfer_in = clock.stop();
 
        // Start time
        clock.start();
        run_kernel2(device_filter, dimension, device_input, device_output, gpu_output_img.width, gpu_output_img.height);
        // End time
        computation = clock.stop();
    
        clock.start(); 
        cudaMemcpy(gpu_output_img.matrix, device_output, size, cudaMemcpyDeviceToHost);
        transfer_out = clock.stop();
 
        print_run(time_cpu, 2, computation*1000, transfer_in*1000, transfer_out*1000); 
        save_pgm_to_file(gpu_file2.c_str(), &gpu_output_img);
        cudaFree(device_filter);
        cudaFree(device_input);
        cudaFree(device_output);
        destroy_pgm_image(&gpu_output_img);
    }
    /*-------------------- Kernel 3 ----------------------*/
    {
        std::string gpu_file3 = "3" + base_gpu_output_filename;
        pgm_image gpu_output_img;
        copy_pgm_image_size(&source_img, &gpu_output_img);
        int32_t *device_input = NULL;
        int32_t *device_output = NULL;
        int8_t *device_filter = NULL;
        size_t size = gpu_output_img.width * gpu_output_img.height * sizeof(int32_t);
        cudaMalloc((void **)&device_filter, dimension*dimension*sizeof(int8_t));
        cudaMalloc((void **)&device_input, size);
        cudaMalloc((void **)&device_output, size);
        float transfer_in, transfer_out, computation;
 
        Clock clock;
        clock.start();
        cudaMemcpy(device_filter, lp5_m, dimension*dimension*sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(device_input, source_img.matrix, size, cudaMemcpyHostToDevice);
        cudaMemcpy(device_output, gpu_output_img.matrix, size, cudaMemcpyHostToDevice);
        transfer_in = clock.stop();
 
        // Start time
        clock.start();
        run_kernel3(device_filter, dimension, device_input, device_output, gpu_output_img.width, gpu_output_img.height);
        // End time
        computation = clock.stop();
    
        clock.start(); 
        cudaMemcpy(gpu_output_img.matrix, device_output, size, cudaMemcpyDeviceToHost);
        transfer_out = clock.stop();
 
        print_run(time_cpu, 3, computation*1000, transfer_in*1000, transfer_out*1000); 
        save_pgm_to_file(gpu_file3.c_str(), &gpu_output_img);
        cudaFree(device_filter);
        cudaFree(device_input);
        cudaFree(device_output);
        destroy_pgm_image(&gpu_output_img);
    }
    /*-------------------- Kernel 4 ----------------------*/
    {
        std::string gpu_file4 = "4" + base_gpu_output_filename;
        pgm_image gpu_output_img;
        copy_pgm_image_size(&source_img, &gpu_output_img);
        int32_t *device_input = NULL;
        int32_t *device_output = NULL;
        int8_t *device_filter = NULL;
        size_t size = gpu_output_img.width * gpu_output_img.height * sizeof(int32_t);
        cudaMalloc((void **)&device_filter, dimension*dimension*sizeof(int8_t));
        cudaMalloc((void **)&device_input, size);
        cudaMalloc((void **)&device_output, size);
        float transfer_in, transfer_out, computation;
 
        Clock clock;
        clock.start();
        cudaMemcpy(device_filter, lp5_m, dimension*dimension*sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(device_input, source_img.matrix, size, cudaMemcpyHostToDevice);
        cudaMemcpy(device_output, gpu_output_img.matrix, size, cudaMemcpyHostToDevice);
        transfer_in = clock.stop();
 
        // Start time
        clock.start();
        run_kernel4(device_filter, dimension, device_input, device_output, gpu_output_img.width, gpu_output_img.height);
        // End time
        computation = clock.stop();
    
        clock.start(); 
        cudaMemcpy(gpu_output_img.matrix, device_output, size, cudaMemcpyDeviceToHost);
        transfer_out = clock.stop();
 
        print_run(time_cpu, 4, computation*1000, transfer_in*1000, transfer_out*1000); 
        save_pgm_to_file(gpu_file4.c_str(), &gpu_output_img);
        cudaFree(device_filter);
        cudaFree(device_input);
        cudaFree(device_output);
        destroy_pgm_image(&gpu_output_img);
    }
    /*-------------------- Kernel 5 ----------------------*/
    {   
        std::string gpu_file5 = "5" + base_gpu_output_filename;
        pgm_image gpu_output_img;
        copy_pgm_image_size(&source_img, &gpu_output_img);
        int32_t *device_input = NULL;
        int32_t *device_output = NULL;
        int8_t *device_filter = NULL;
        size_t size = gpu_output_img.width * gpu_output_img.height * sizeof(int32_t);
        cudaMalloc((void **)&device_filter, dimension*dimension*sizeof(int8_t));
        cudaMalloc((void **)&device_input, size);
        cudaMalloc((void **)&device_output, size);
        float transfer_in, transfer_out, computation;
 
        Clock clock;
        clock.start();
        cudaMemcpy(device_filter, lp5_m, dimension*dimension*sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(device_input, source_img.matrix, size, cudaMemcpyHostToDevice);
        cudaMemcpy(device_output, gpu_output_img.matrix, size, cudaMemcpyHostToDevice);
        transfer_in = clock.stop();
 
        // Start time
        clock.start();
        run_kernel5(device_filter, dimension, device_input, device_output, gpu_output_img.width, gpu_output_img.height);
        // End time
        computation = clock.stop();
    
        clock.start(); 
        cudaMemcpy(gpu_output_img.matrix, device_output, size, cudaMemcpyDeviceToHost);
        transfer_out = clock.stop();
 
        print_run(time_cpu, 5, computation*1000, transfer_in*1000, transfer_out*1000); 
        save_pgm_to_file(gpu_file5.c_str(), &gpu_output_img);
        cudaFree(device_filter);
        cudaFree(device_input);
        cudaFree(device_output);
        destroy_pgm_image(&gpu_output_img);
    }


}
