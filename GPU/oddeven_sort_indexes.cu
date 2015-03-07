/*
 * oddeven_sort.cu
 *
 *  Created on: Mar 7, 2015
 *      Author: nicolas.legroux
 */

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "../utilities.h"
#include "../configuration.h"

#define BLOCK_SIZE 218

__global__ void oddeven_step(double *data, int* indexes, int n, int step){

	int index_thread = blockDim.x * blockIdx.x + threadIdx.x;

	if(step%2 == 0 && index_thread%2 == 0 && (index_thread+1 < n) ){

		double data_left = data[index_thread];
		double data_right = data[index_thread+1];

		if(data_left > data_right){
			int index_left = indexes[index_thread];
			data[index_thread] = data_right;
			data[index_thread+1] = data_left;
			indexes[index_thread] = indexes[index_thread+1];
			indexes[index_thread+1] = index_left;
		}
	}

	else if(step%2 == 1 && index_thread%2 == 1 && (index_thread+1 < n)){
		double data_left = data[index_thread];
		double data_right = data[index_thread+1];

		if(data_left > data_right){
			int index_left = indexes[index_thread];
			data[index_thread] = data_right;
			data[index_thread+1] = data_left;
			indexes[index_thread] = indexes[index_thread+1];
			indexes[index_thread+1] = index_left;
		}
	}

	__syncthreads();
}

void oddeven_sort_indexes(double *data, int * indexes, int n){

	double* d_data;
	int* d_indexes;

	int data_size = n*sizeof(double);
	int indexes_size = n*sizeof(int);

	checkCudaErrors(cudaMalloc((void**)&d_data, data_size));
	checkCudaErrors(
				cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&d_indexes, indexes_size));
	checkCudaErrors(
				cudaMemcpy(d_indexes, indexes, indexes_size, cudaMemcpyHostToDevice));

	int dim_block = BLOCK_SIZE;
	int dim_grid = n/BLOCK_SIZE;

	if(n%BLOCK_SIZE != 0){
		dim_grid++;
	}

	for(int i=0; i < n; i++){
		oddeven_step<<<dim_grid, dim_block>>>(d_data, d_indexes, n, i);
	}

	checkCudaErrors(
			cudaMemcpy(indexes, d_indexes, indexes_size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_indexes));
}

void test_oddeven_sort(){

	int DATA_SIZE = 10;

	double * data = new double[DATA_SIZE];
	array_fill(data, N);

	int * indexes = new int[DATA_SIZE];
	for(int i=0; i<DATA_SIZE; i++){
		indexes[i] = i;
	}

	oddeven_sort_indexes(data, indexes, DATA_SIZE);

	printf("\nShowing output of not sorted : \n");

	for(int i=0; i<N; i++){
		printf("%1.5f ", data[i]);
	}

	printf("\nShowing output of sort : \n");

	for(int i=0; i<N; i++){
		printf("%1.5f ", data[indexes[i]]);
	}

	printf("\n");
}

