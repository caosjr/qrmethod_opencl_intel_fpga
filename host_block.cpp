#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#define TAM 47
#define BLOCK_SIZE 65
#define PROGRAM_NAME "qr_method_original_sqrt_block_single_cache_v3.aocx"
#define KERNEL_FUNC "qr_method"

using namespace aocl_utils;

/**
 * * Specific functions definition in OpenCL
 * */
extern "C"
{
bool init_opencl_data_structure_dense_(int *n, bool *success);
void init_structure_single_dense (int *n, double dataQ[], double dataR[]);
int qr_method_fpga(int *n);
void free_resources();
static void dump_error (const char *str, cl_int status);
}


char dname[500]; //name of the platform

/**
 * * OpenCL common variables
 * */
cl_uint num_devices, num_platforms;
cl_platform_id platform = NULL;
cl_device_id device;
cl_context context = NULL;
cl_kernel kernel;
cl_command_queue queue;
cl_program program = NULL;
cl_event computation_event, send_event, receive_event;
double *input_q_buf, *input_r_buf; //pointers to buffers

void print_matrix(double *matrix){
    for (int i = 0; i < TAM; i++){
        for (int j = 0; j < TAM; j++) {
            printf("%lf\t", matrix[i*TAM + j]);
        }
        printf("\n");
    }
}

void print_vector(double *vector){
    for (int i = 0; i < TAM; i++){
        printf("%lf\t", vector[i]);
    }
}

bool init_opencl_data_structure_dense_(int *n, bool *success){
    cl_int status;
    int ARRAY_SIZE = *n;

    // get the platform ID
    status = clGetPlatformIDs(1, &platform, &num_platforms);
    if(status != CL_SUCCESS) {
        dump_error("Failed clGetPlatformIDs.", status);
        free_resources();
        return 1;
    }

    if(num_platforms != 1) {
        printf("Found %d platforms!\n", num_platforms);
        free_resources();
        return 1;
    }

    // get the device ID
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);
    if(status != CL_SUCCESS) {
        dump_error("Failed clGetDeviceIDs.", status);
        free_resources();
        return 1;
    }

    if(num_devices != 1) {
        printf("Found %d devices!\n", num_devices);
        free_resources();
        return 1;
    }

    // create a context
    context = clCreateContext(0, 1, &device, NULL, NULL, &status);
    if(status != CL_SUCCESS) {
        dump_error("Failed clCreateContext.", status);
        free_resources();
        return 1;
    }

    input_q_buf = (double*)clSVMAllocAltera(context, 0, BLOCK_SIZE*ARRAY_SIZE*ARRAY_SIZE * sizeof (double), 1024);
    input_r_buf = (double*)clSVMAllocAltera(context, 0, BLOCK_SIZE*ARRAY_SIZE*ARRAY_SIZE * sizeof (double), 1024);

    if(!input_q_buf || !input_r_buf) {
        dump_error("Failed to allocate buffers.", status);
        free_resources();
        return 1;
    }

    // create a command queue
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    if(status != CL_SUCCESS) {
        dump_error("Failed clCreateCommandQueue.", status);
        free_resources();
        return 1;
    }

    // create the program
    cl_int kernel_status;

    size_t binsize = 0;
    unsigned char * binary_file = loadBinaryFile(PROGRAM_NAME, &binsize); // Colocar aqui o nome do arquivo Kernel

    if(!binary_file) {
        dump_error("Failed loadBinaryFile.", status);
        free_resources();
        return 1;
    }
    //printf("\nLoad binary\n");

    program = clCreateProgramWithBinary(context, 1, &device, &binsize, (const unsigned char**)&binary_file, &kernel_status, &status);
    if(status != CL_SUCCESS) {
        dump_error("Failed clCreateProgramWithBinary.", status);
        free_resources();
        return 1;
    }

    delete [] binary_file;
    // build the program
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    if(status != CL_SUCCESS) {
        dump_error("Failed clBuildProgram.", status);
        free_resources();
        return 1;
    }

    // create the kernel
    kernel = clCreateKernel(program, KERNEL_FUNC, &status);
    if(status != CL_SUCCESS) {
        dump_error("Failed clCreateKernel.", status);
        free_resources();
        return 1;
    }

    //This variable is not mandatory, I kept because I needed it for another project
    *success = true;
    return true;
}

void init_structure_single_dense (int *n, double dataQ[], double dataR[]) {
    int ARRAY_SIZE = *n;
    if(num_devices == 0) {
        checkError(-1, "No devices");
    }

    memcpy(input_q_buf, dataQ, BLOCK_SIZE*ARRAY_SIZE*ARRAY_SIZE * sizeof(double));
    memcpy(input_r_buf, dataR, BLOCK_SIZE*ARRAY_SIZE*ARRAY_SIZE * sizeof(double));
}

int qr_method_fpga (int *n){
    cl_int status;
    int ARRAY_SIZE = *n;
    struct timeval before, after;

    // set the arguments
    status = clSetKernelArgSVMPointerAltera(kernel, 0, (void*)input_q_buf);
    if(status != CL_SUCCESS) {
        dump_error("Failed set arg 0.", status);
        return 1;
    }

    status = clSetKernelArgSVMPointerAltera(kernel, 1, (void*)input_r_buf);
    if(status != CL_SUCCESS) {
        dump_error("Failed Set arg 1.", status);
        free_resources();
        return 1;
    }

    gettimeofday(&before, NULL);
    status = clEnqueueSVMMap(queue, CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, (void*) input_q_buf, BLOCK_SIZE * ARRAY_SIZE * ARRAY_SIZE * sizeof (double), 0, NULL, NULL);
    if(status != CL_SUCCESS) {
        dump_error("Failed clEnqueueSVMMap", status);
        free_resources();
        return 1;
    }

    status = clEnqueueSVMMap(queue, CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, (void *) input_r_buf, BLOCK_SIZE * ARRAY_SIZE * ARRAY_SIZE * sizeof (double), 0, NULL, &send_event);
    if(status != CL_SUCCESS) {
        dump_error("Failed clEnqueueSVMMap", status);
        free_resources();
        return 1;
    }
    clWaitForEvents(1, &send_event);
    clReleaseEvent(send_event);
    gettimeofday(&after, NULL);
    printf("Send: %ld\n", ((after.tv_sec * 1000000 + after.tv_usec) - (before.tv_sec * 1000000 + before.tv_usec)));

    // execução kernel
    gettimeofday(&before, NULL);
    status = clEnqueueTask(queue, kernel, 0, NULL, &computation_event);
    clWaitForEvents(1, &computation_event);
    clReleaseEvent(computation_event);
    gettimeofday(&after, NULL);
    printf("Solve: %ld\n", ((after.tv_sec * 1000000 + after.tv_usec) - (before.tv_sec * 1000000 + before.tv_usec)));

    if (status != CL_SUCCESS) {
        dump_error("Failed to launch kernel.", status);
        free_resources();
        return 1;
    }

    gettimeofday(&before, NULL);
    status = clEnqueueSVMUnmap(queue, (void *)input_q_buf, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
        dump_error("Failed clEnqueueSVMUnmap", status);
        free_resources();
        return 1;
    }

    status = clEnqueueSVMUnmap(queue, (void *)input_r_buf, 0, NULL, &receive_event);
    if(status != CL_SUCCESS) {
        dump_error("Failed clEnqueueSVMUnmap", status);
        free_resources();
        return 1;
    }
    clWaitForEvents(1, &receive_event);
    clReleaseEvent(receive_event);
    gettimeofday(&after, NULL);
    printf("Receive: %ld\n", ((after.tv_sec * 1000000 + after.tv_usec) - (before.tv_sec * 1000000 + before.tv_usec)));

    clFinish(queue);
    return 0;
}


static void dump_error(const char *str, cl_int status) {
    printf("%s\n", str);
    printf("Error code: %d\n", status);
}

// free the resources allocated during initialization
void free_resources() {
    if(kernel)
        clReleaseKernel(kernel);
    if(program)
        clReleaseProgram(program);
    if(queue)
        clReleaseCommandQueue(queue);
    if(input_q_buf)
        clSVMFreeAltera(context,input_q_buf);
    if(input_r_buf)
        clSVMFreeAltera(context,input_r_buf);
    if(context)
        clReleaseContext(context);
}

int main() {
    double dataA[TAM*TAM] =
            {0.008356, -0.000000, 0.000020, -0.000020, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000001, -0.000000, -0.000001, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             0.007629, -0.000000, 0.015962, -0.007629, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, 0.000000, 0.000000, -0.000000, -0.000000, 0.000000, 0.000000, 0.000000, -0.000000,
             0.000013, -0.000000, -0.000000, 0.008346, -0.000013, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, 0.000000, -0.000000, 0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, 0.032613, -0.046712, 0.155322, -0.018533, -0.000000, -0.000073, -0.000000, -0.000000, -0.000000, -0.000030, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000026, -0.000000, -0.000000, -0.000000, 0.000138, 0.095613, 0.000000, 0.000023, 0.000045, -0.000000, -0.087712, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, 0.000000, 0.000000, -0.000000, -0.000000, 0.000000, -0.000047, -0.095742, -0.000001,
             -0.000000, -0.000000, -0.000000, -0.023654, -0.023654, 0.031988, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, -0.048183, -0.000000, -0.000000, -0.000000, -0.000000, 0.056517, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.048183, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -77436.298921, -0.000000, -0.138916, 0.162288, -0.023372, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.065714, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 77443.051476, -0.000000, -0.131427, -1.839983, -0.000000, -0.000000, -0.000000, -6.036983, 6.571366, -0.000000, -0.328568, -0.000000, -0.000000, -0.871035, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.985705,
             -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -719149154.597408, 840256257.842559, -242214206.473636, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             0.029333, 0.000000, 0.006097, 0.171589, -0.000533, -0.000000, -0.006097, -0.171276, 0.000000, 0.000000, -0.000000, 0.124019, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 14.142115, -1.261046, 0.229139, 0.001837, 0.716671, 0.167367, 11.272654, 0.007067, 0.443049, 0.299556, -0.005173, 0.015943, 0.000000, 0.000220, 0.000000, 0.000000, -0.006292, -0.000000, -0.229139, -0.631153, -0.171600, -11.272654, -0.000020, -0.006640, -0.000000, -0.393519, -0.000000, -0.003725,
             0.000905, -0.000000, 0.010573, 0.010287, 0.000000, -0.000000, -0.000000, -0.000000, -0.020860, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.011478, 0.040672, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
             -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             0.000002, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000001, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000001, -0.000000, -0.000000, -0.000000, -0.000000, 0.008335, -0.000000, -0.000000, -0.000001, -0.000001, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             0.000005, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000002, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000001, -0.000002, -0.000000, -0.000000, -0.000000, -0.000002, 0.008339, -0.000000, -0.000005, -0.000000, -0.000000, -0.000002, -0.000000, -0.000000, -0.000000, -0.000000, -0.000001, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000001, -0.000000, -0.000001,
             -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             0.000000, -0.000000, -0.000000, -0.000215, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.008549, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000215, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.008333, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, 0.009451, -0.009451, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.009451, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.009451, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.017785, 0.000000, 0.000000, 0.000000, -0.000000, -0.000000, 0.000000, 0.000000, 0.000000, -0.000000,
             -0.000000, -0.000000, 0.005308, -0.004859, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.003942, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000159, -0.001759, -0.002895, -0.000181, -0.000449, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000479, 0.013207, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000690,
             -0.000000, -0.000000, 0.008437, -0.008437, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.008437, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.011801, -0.003554, -0.000440, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.016770, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, 0.004870, -0.004125, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.004125, -0.000000, -0.000000, -0.000000, -0.001842, -0.000000, -0.000000, -0.002951, -0.000000, -0.000000, -0.002214, -0.000745, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.013203, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, -0.000000, 0.288037, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.030734, -0.000000, -0.000000, -0.000000, -0.000000, -0.288037, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.296370, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000,
             12.694951, -0.000000, -0.000000, 0.518466, -0.000000, -0.000000, -0.518466, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -12.694951, -57.897810, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -71.111227, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 2908.112258, -2836.992698, -0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, 0.004870, -0.004632, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.004632, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.010080, -0.000238, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.013203, 0.000000, -0.000000, -0.000000,
             -0.000000, -0.000000, 0.014853, 0.113279, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.001832, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000869, -0.001094, -0.000000, -0.001288, -0.000000, -0.128132, -0.000000, -0.000000, -0.000000, -0.000000, -0.011605, 0.000000, 0.000000, 0.000000, -0.000000, -0.000000, 0.000000, 0.149903, 0.000000, -0.000381,
             -0.000000, -0.000000, 0.004870, -0.008844, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000896, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.001140, -0.004927, -0.001844, -0.000000, -0.000896, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, 0.013203, -0.000000,
             -0.000000, -0.000000, 0.004870, -0.004870, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, 0.013203
            };

    int i = 0, size = TAM;
    unsigned int offset = 0;
    bool success = false;
    double dataB[TAM] = {7,4,0,3,6,9,0,3,3,8,0,8,0,1,4,6,7,9,2,4,6,9,8,7,3,4,7,4,1,7,1,3,1,1,2,4,0,6,5,1,9,4,1,9,2,9,9};
    double *dataR = (double*) malloc(BLOCK_SIZE*TAM*TAM*sizeof(double));
    double *dataQ = (double*) malloc(BLOCK_SIZE*TAM*TAM*sizeof(double));
    struct timeval before, after;
    for (i = 0; i < BLOCK_SIZE; i++){
        memcpy(dataQ + offset, dataA, sizeof(dataA));
        offset += TAM*TAM;
    }
    memset(dataR, 0, sizeof(dataR));
    /*printf("\n\nMatriz A\n");
    print_matrix(dataA);
    printf("\n\nMatriz Q\n");
    print_matrix(dataQ);
    printf("\n\nMatriz R\n");
    print_matrix(dataR);*/


    init_opencl_data_structure_dense_(&size, &success);
    init_structure_single_dense(&size, dataQ, dataR);
    run_single_dense(&size);

    free(dataQ);
    free(dataR);
    free_resources();

    return 0;
}
