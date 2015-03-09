/*
 * extract_minimums.cu
 *
 *  Created on: Mar 8, 2015
 *      Author: nicolas.legroux
 */

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "extract_minimums.h"
#include "../utilities.h"

#define BLOCK_DIM_X 512
#define MAX_DOUBLE 10000000000000.0;

__global__ void set_maximum_double_value(double * data, int * indexes_to_reset, int n_train, int n_test){
	unsigned int thread_global_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(thread_global_idx < n_test){
		int index_to_reset = indexes_to_reset[thread_global_idx];
		data[n_train * thread_global_idx + index_to_reset] = MAX_DOUBLE;
	}
}

__global__ void find_minimum(double * data, int * indexes, int * output_indexes,
		int n_train, int n_test) {
	__shared__ double block_data[BLOCK_DIM_X];
	__shared__ int block_indexes[BLOCK_DIM_X];

	unsigned int thread_block_idx = threadIdx.x;
	unsigned int thread_global_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int thread_global_idy = blockIdx.y * blockDim.y ;

	if (thread_global_idx < n_train) {
		block_data[thread_block_idx] = data[thread_global_idy * n_train + thread_global_idx];
		block_indexes[thread_block_idx] = indexes[thread_global_idy * n_train + thread_global_idx];
	} else {
		block_data[thread_block_idx] = MAX_DOUBLE
		block_indexes[thread_block_idx] = -1;
	}
	__syncthreads();

	for (unsigned int s = 1; s < blockDim.x; s *= 2) {

		if (thread_block_idx % (2 * s) == 0) {
			if (block_data[thread_block_idx + s]
					< block_data[thread_block_idx]) {
				block_data[thread_block_idx] = block_data[thread_block_idx + s];
				block_indexes[thread_block_idx] = block_indexes[thread_block_idx
						+ s];
			}
		}
	}

	if (thread_block_idx == 0) {
		output_indexes[thread_global_idy * gridDim.x + blockIdx.x] = block_indexes[0];
	}
	__syncthreads();
}

void find_k_minimums(double * data, int n_train, int n_test, int k, int * k_minimum_indexes) {
	double* d_data;
	int* d_indexes;
	int* d_indexes_to_reset;
	int* output_indexes;
	int* host_indexes_cpy;

	int data_size = n_train * n_test * sizeof(double);
	int indexes_size = n_train * n_test * sizeof(int);

	int * indexes = (int*) malloc(n_train * n_test * sizeof(int));

	for(int i=0; i<n_test; i++){
		for(int j=0; j < n_train; j++){
			indexes[i*n_train+j] = j;
		}
	}

	checkCudaErrors(cudaMalloc((void**) &d_data, data_size));
	checkCudaErrors(
			cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**) &d_indexes, indexes_size));
	checkCudaErrors(
			cudaMemcpy(d_indexes, indexes, indexes_size, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**) &d_indexes_to_reset, n_test*sizeof(int)));

	int dim_grid_y = n_test;
	int dim_block_y = 1;

	int dim_block_x =  BLOCK_DIM_X;

	int dim_grid_x = n_train / dim_block_x;

	if (n_train % BLOCK_DIM_X != 0) {
		dim_grid_x++;
	}

	dim3 dim_grid(dim_grid_x, dim_grid_y);
	dim3 dim_block(dim_block_x, dim_block_y);

	checkCudaErrors(cudaMalloc((void**) &output_indexes, dim_grid.x * n_test * sizeof(int)));
	host_indexes_cpy = (int*) malloc(dim_grid.x * n_test * sizeof(int));
	int * index_minimums = (int*) malloc(n_test * sizeof(int));

	int dim_block_x_reset = BLOCK_DIM_X/4;
	int dim_grid_x_reset = n_test / dim_block_x_reset;

	if(n_test % dim_block_x_reset != 0){
		dim_grid_x_reset++;
	}

	for(int j=0; j<k; j++){
		find_minimum<<<dim_grid, dim_block>>>(d_data, d_indexes, output_indexes, n_train, n_test);
		checkCudaErrors(cudaMemcpy(host_indexes_cpy, output_indexes, dim_grid.x* n_test * sizeof(int), cudaMemcpyDeviceToHost));

		for(int i=0; i<n_test; i++){
			index_minimums[i] = find_minimum_cpu((data+i*n_train), (host_indexes_cpy + i*dim_grid.x), dim_grid.x);
			k_minimum_indexes[i*k+j] = index_minimums[i];
		}

		checkCudaErrors(
					cudaMemcpy(d_indexes_to_reset, index_minimums, n_test*sizeof(int), cudaMemcpyHostToDevice));

		set_maximum_double_value<<<dim_grid_x_reset, dim_block_x_reset>>>(d_data, d_indexes_to_reset, n_train, n_test);

		cudaError_t err = cudaThreadSynchronize();
		if(err != cudaSuccess){
		    printf("cudaThreadSynchronize error: %s\n", cudaGetErrorString(err));
		    exit(1);
		}
	}

	free(host_indexes_cpy);
	free(index_minimums);

	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_indexes));
	checkCudaErrors(cudaFree(output_indexes));
	checkCudaErrors(cudaFree(d_indexes_to_reset));
}

int find_minimum_cpu(double * data, int * indexes, int n){
	int index_min = indexes[0];
	double min_value = data[index_min];

	for(int i=1; i< n; i++){
		if(data[indexes[i]] < min_value){
			min_value = data[indexes[i]];
			index_min = indexes[i];
		}
	}

	return index_min;
}

void test_extract_minimum(int n_train, int n_test, int k){
	double * data = new double[n_train * n_test];

	int * k_minimum_indexes = (int*) malloc(k*n_test*sizeof(int));

	array_fill(data, n_train * n_test);
	//print_vectors_in_row_major_order(data, n_test, n_train);

	find_k_minimums(data, n_train, n_test, k, k_minimum_indexes);

	//print_vectors_in_row_major_order(k_minimum_indexes, n_test, k);

	delete[] data;
	delete[] k_minimum_indexes;
}

