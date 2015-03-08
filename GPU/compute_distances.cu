#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <helper_functions.h>
#include <stdlib.h>
#include <stdio.h>
#include "compute_distances.h"
#include "../utilities.h"

#define MAX_THREADS_PER_BLOCK 1024
#define SIMPLE_BLOCK_SIZE 1024
#define BLOCK_DIM 32

__global__ void gpu_distance_withreduction(double* data,
		double* point, int n, int dim, double* distance) {

	extern __shared__ double distComponents[];

	int shift_dim = threadIdx.x;
	int shift_point = blockIdx.x * blockDim.y + threadIdx.y;
	int shift_point_in_block = threadIdx.y;

	if (shift_dim < dim  && shift_point < n) {
		double d = 0;
		d = abs(data[shift_point * dim + shift_dim] - point[shift_dim]);
		distComponents[shift_point_in_block * blockDim.y + shift_dim] = d * d;
	}

	__syncthreads();

	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {

		if (shift_dim < offset) {
			distComponents[shift_point_in_block * blockDim.y + shift_dim] +=
					distComponents[shift_point_in_block * blockDim.y + shift_dim
							+ offset];
		}

		__syncthreads();
	}

	if (shift_dim == 0 && shift_point < n) {
		distance[shift_point] =
				distComponents[shift_point_in_block * blockDim.y];
	}
}

__global__ void gpu_distance(double* data, double* point,
		int n, int dim, double* distance) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n)
		return;

	double d = 0;

	for (int j = 0; j < dim; j++) {
		double temp = abs(data[i * dim + j] - point[j]);
		d += temp * temp;
	}

	distance[i] = d;
}

//blockDim.x must be larger than Dim
//BlockDim.y should be equal to one
//GridDim.y should be equal to the number of training points
//test_data should be stored in row-major order
//train_data should be stored in column-major order
__global__ void gpu_distances(double* train_data, double * test_data, int dim,
		int n_train, int n_test, double * distances) {

	int t_idx_in_block = threadIdx.x;
	int t_idx_global = blockDim.x * blockIdx.x + threadIdx.x;
	int t_idy_global = blockIdx.y;

	extern __shared__ double test_data_point[];

	//Load one data point into shared memory
	if (t_idx_in_block < dim) {
		test_data_point[t_idx_in_block] = test_data[t_idy_global * dim
				+ t_idx_in_block];
	}

	__syncthreads();

	//Now compute distance
	double dist = 0;
	if (t_idx_global < n_train) {
		for (int i = 0; i < dim; i++) {
			double temp = train_data[i * n_train + t_idx_global]
					- test_data_point[i];
			dist += temp * temp;
		}

		distances[t_idy_global * n_train + t_idx_global] = dist;
	}
}

//All The data is assumed to be in row major order
/* At the end the distance matrix is as follows :
 *  [ distance(test1, train1)    distance(test2, train1)   distances(test3, train1) ...
 *   distance(test1, train2     ...
 *  ...																					]
 */
void gpu_compute_distances(double *train_data, double *test_data, int n_train,
		int n_test, int dim, double* distances) {
	double *train_data_copy = new double[n_train * dim];
	array_copy(train_data, train_data_copy, n_train * dim);
	convert_row_major_to_column_major(train_data_copy, n_train, dim);

	double* d_train_data;
	double* d_test_data;
	double* d_distances;

	checkCudaErrors(
			cudaMalloc((void**)&d_train_data, n_train*dim*sizeof(double)));
	checkCudaErrors(
			cudaMalloc((void**)&d_test_data, n_test*dim*sizeof(double)));
	checkCudaErrors(
			cudaMalloc((void**)&d_distances, n_train*n_test*sizeof(double)));

	checkCudaErrors(
			cudaMemcpy(d_train_data, train_data_copy, n_train*dim*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(d_test_data, test_data, n_test*dim*sizeof(double), cudaMemcpyHostToDevice));

	int dim_grid_y = n_test;
	int dim_block_x = multiple_of_32(dim);
	int dim_block_y = 1;
	int dim_grid_x = n_train / dim_block_x;

	if (n_train % dim_block_x != 0)
		dim_grid_x++;

	dim3 dim_grid(dim_grid_x, dim_grid_y);
	dim3 dim_block(dim_block_x, dim_block_y);

	gpu_distances<<<dim_grid, dim_block, dim * sizeof(double)>>>(d_train_data,
			d_test_data, dim, n_train, n_test, d_distances);

	checkCudaErrors(
			cudaMemcpy(distances, d_distances, n_train*n_test*sizeof(double), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_train_data));
	checkCudaErrors(cudaFree(d_test_data));
	checkCudaErrors(cudaFree(d_distances));

	delete[] train_data_copy;
}

void gpu_compute_distance(double* train_data, double* test_point, int n_train, int dim, double* distance) {

	int datasize = n_train * dim * sizeof(double);

	int nblock = n_train / SIMPLE_BLOCK_SIZE;
	if (n_train % SIMPLE_BLOCK_SIZE != 0)
		nblock += 1;
	int nthread = SIMPLE_BLOCK_SIZE;

	double *d_data;
	double *d_point;
	double *d_distance;

	printf("\nGrid dimension : %d\n", nblock);
	printf("Block dimension : %d\n", nthread);

	checkCudaErrors(cudaMalloc((void**)&d_data, datasize));
	checkCudaErrors(cudaMalloc((void**)&d_distance, n_train*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_point, dim*sizeof(double)));

	checkCudaErrors(
			cudaMemcpy(d_data, train_data, datasize, cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(d_point, test_point, dim*sizeof(double), cudaMemcpyHostToDevice));

	gpu_distance<<<nblock, nthread>>>(d_data, d_point, n_train, dim, d_distance);

	checkCudaErrors(
			cudaMemcpy(distance, d_distance, n_train*sizeof(double), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_distance));
	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_point));
}

void gpu_compute_distance_withreduction(double* train_data, double* test_point, int n_train, int dim,
		double* distance) {
	int datasize = n_train * dim * sizeof(double);

	int block_dim_x = multiple_of_32(dim);
	int block_dim_y = MAX_THREADS_PER_BLOCK / block_dim_x;

	int nblock = n_train / block_dim_y;
	if (n_train % block_dim_y != 0)
		nblock++;

	dim3 dim_block(block_dim_x, block_dim_y, 1);

	printf("\nGrid dimension : %d\n", nblock);
	printf("Block dimension : %d * %d\n", dim_block.x, dim_block.y);

	double *d_data;
	double *d_point;
	double *d_distance;

	checkCudaErrors(cudaMalloc((void**)&d_data, datasize));
	checkCudaErrors(cudaMalloc((void**)&d_distance, n_train*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_point, dim*sizeof(double)));

	checkCudaErrors(
			cudaMemcpy(d_data, train_data, datasize, cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(d_point, test_point, dim*sizeof(double), cudaMemcpyHostToDevice));

	gpu_distance_withreduction<<<nblock, dim_block,
			block_dim_y * block_dim_x * sizeof(double)>>>(d_data, d_point, n_train, dim, d_distance);

	checkCudaErrors(
			cudaMemcpy(distance, d_distance, n_train*sizeof(double), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_distance));
	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_point));
}

void test_distances_multiple() {
	int n_train = 3;
	int n_test = 4;
	int dim = 2;

	double *train_data = new double[n_train * dim];
	double *test_data = new double[n_test * dim];
	array_fill(train_data, n_train * dim);
	array_fill(test_data, n_test * dim);

	printf("\n-------- Printing the TRAIN data ---------\n");
	print_vectors_in_row_major_order(train_data, n_train, dim);

	printf("\n-------- Printing the TEST data ---------\n");
	print_vectors_in_row_major_order(test_data, n_test, dim);

	double *distances = new double[n_train * n_test];

	gpu_compute_distances(train_data, test_data, n_train, n_test, dim,
			distances);

	printf("\n-------- Printing the DISTANCE matrix ---------\n");
	print_vectors_in_row_major_order(distances, n_test, n_train);
}




