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

#define BLOCK_SIZE 512

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

//n_test should be equal to to the y dimension of the grid
//The y dimension of a block is 1
__global__ void oddeven_step_multiple(double *distances, int* indexes, int n_train, int n_test, int step){
	int id_array_to_be_sorted = blockIdx.y;
	int id_thread_x = blockDim.x * blockIdx.x + threadIdx.x;

	if(step%2 == 0 && id_thread_x%2 == 0 && (id_thread_x+1 < n_train)){
		double data_left = distances[id_array_to_be_sorted*n_train + id_thread_x];
		double data_right = distances[id_array_to_be_sorted*n_train + id_thread_x + 1];

		if(data_left > data_right){
			int index_left = indexes[id_array_to_be_sorted*n_train + id_thread_x];
			distances[id_array_to_be_sorted*n_train + id_thread_x] = data_right;
			distances[id_array_to_be_sorted*n_train + id_thread_x+1] = data_left;
			indexes[id_array_to_be_sorted*n_train + id_thread_x] = indexes[id_array_to_be_sorted*n_train + id_thread_x+1];
			indexes[id_array_to_be_sorted*n_train + id_thread_x+1] = index_left;
		}
	}

	else if(step%2 == 1 && id_thread_x%2 == 1 && (id_thread_x+1 < n_train)){
		double data_left = distances[id_array_to_be_sorted*n_train + id_thread_x];
		double data_right = distances[id_array_to_be_sorted*n_train + id_thread_x + 1];

		if(data_left > data_right){
			int index_left = indexes[id_array_to_be_sorted*n_train + id_thread_x];
			distances[id_array_to_be_sorted*n_train + id_thread_x] = data_right;
			distances[id_array_to_be_sorted*n_train + id_thread_x+1] = data_left;
			indexes[id_array_to_be_sorted*n_train + id_thread_x] = indexes[id_array_to_be_sorted*n_train + id_thread_x+1];
			indexes[id_array_to_be_sorted*n_train + id_thread_x+1] = index_left;
		}
	}
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

void oddeven_sort_indexes_multiple(double *distances, int * indexes, int n_train, int n_test){

	double* d_distances;
	int* d_indexes;

	int data_size = n_train*n_test*sizeof(double);
	int indexes_size = n_train*n_test*sizeof(int);

	checkCudaErrors(cudaMalloc((void**)&d_distances, data_size));
	checkCudaErrors(
				cudaMemcpy(d_distances, distances, data_size, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&d_indexes, indexes_size));
	checkCudaErrors(
				cudaMemcpy(d_indexes, indexes, indexes_size, cudaMemcpyHostToDevice));

	int dim_grid_y = n_test;
	int dim_block_y = 1;
	int dim_block_x = 512;
	int dim_grid_x = n_train/dim_block_x;

	if(n_train%dim_block_x != 0){
		dim_grid_x++;
	}

	dim3 dim_grid(dim_grid_x, dim_grid_y);
	dim3 dim_block(dim_block_x, dim_block_y);

	for(int i=0; i < n_train; i++){
		oddeven_step_multiple<<<dim_grid, dim_block>>>(d_distances, d_indexes, n_train, n_test, i);
	}

	checkCudaErrors(
			cudaMemcpy(indexes, d_indexes, indexes_size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_distances));
	checkCudaErrors(cudaFree(d_indexes));
}

void test_oddeven_sort(int DATA_SIZE){

	double * data = new double[DATA_SIZE];
	array_fill(data, DATA_SIZE);

	int * indexes = new int[DATA_SIZE];
	for(int i=0; i<DATA_SIZE; i++){
		indexes[i] = i;
	}

	oddeven_sort_indexes(data, indexes, DATA_SIZE);

	printf("\nShowing output of not sorted : \n");

	for(int i=0; i<DATA_SIZE; i++){
		printf("%1.5f ", data[i]);
	}

	printf("\nShowing output of sort : \n");

	for(int i=0; i<DATA_SIZE; i++){
		printf("%1.5f ", data[indexes[i]]);
	}

	printf("\n");
}

void test_oddeven_sort_multiple(){

	int n_train = 50;
	int n_test = 3;

	double * distances = new double[n_train*n_test];
	int * indexes = new int[n_train*n_test];

	for(int i=0; i<n_train; i++){
		indexes[i] = i;
		distances[i] = (double)i;
		indexes[n_train+i] = i;
		distances[n_train+i] = (double)(n_train - i);
		indexes[2*n_train+i] = i;
		distances[2*n_train+i] = random_double();
	}

	printf("\n----- Printing the arrays to be sorted : -----\n");
	print_vectors_in_row_major_order(distances, n_test, n_train);
	printf("\n--------------------------------------------- \n");

	oddeven_sort_indexes_multiple(distances, indexes, n_train, n_test);

	printf("\n----- Printing the sorted indexes ----- : \n");
	print_vectors_in_row_major_order(indexes, n_test, n_train);
	printf("\n--------------------------------------------- \n");

	delete[] distances;
	delete[] indexes;
}

