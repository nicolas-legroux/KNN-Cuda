#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <helper_functions.h>
#include <stdlib.h>
#include <stdio.h>
#include "../configuration.h"
#include "gpu_knn.h"

const int blocksize = 16;

#define MAX_THREADS_PER_BLOCK 1024
#define SIMPLE_BLOCK_SIZE 1024

__global__ void gpu_distance_withreduction(double* data, double* distance, double* point,
		int n, int dim) {

	extern __shared__ double distComponents[];

	int shift_dim = threadIdx.x;
	int shift_point = blockIdx.x * blockDim.y + threadIdx.y;
	int shift_point_in_block = threadIdx.y;

	if (shift_dim < DIM && shift_point < n) {
		double d = 0;
		d = abs(data[shift_point * dim + shift_dim] - point[shift_dim]);
		distComponents[shift_point_in_block*blockDim.y+shift_dim] = d * d;
	}

	__syncthreads();

	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {

		if (shift_dim < offset) {
			distComponents[shift_point_in_block*blockDim.y+shift_dim] +=
					distComponents[shift_point_in_block*blockDim.y+shift_dim+offset];
		}

		__syncthreads();
	}

	if (shift_dim == 0 && shift_point < n) {
		distance[shift_point] = distComponents[shift_point_in_block*blockDim.y];
	}
}

__global__ void gpu_distance(double* data, double* distance, double* point, int n,
		int dim) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n)
		return;

	double d = 0;

	for (int j = 0; j < dim; j++){
		double temp = abs(data[i * dim + j] - point[j]);
		d += temp*temp;
	}

	distance[i] = d;
}

void gpu_compute_distance(double* data, double* point, double* distance) {

	int datasize = N * DIM * sizeof(double);

	int nblock = N / SIMPLE_BLOCK_SIZE;
	if(N%SIMPLE_BLOCK_SIZE !=0)
		nblock += 1;
	int nthread = SIMPLE_BLOCK_SIZE;

	double *d_data;
	double *d_point;
	double *d_distance;

	printf("\nGrid dimension : %d\n", nblock);
	printf("Block dimension : %d\n", nthread);

	checkCudaErrors(cudaMalloc((void**)&d_data, datasize));
	checkCudaErrors(cudaMalloc((void**)&d_distance, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_point, DIM*sizeof(double)));

	checkCudaErrors(
			cudaMemcpy(d_data, data, datasize, cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(d_point, point, DIM*sizeof(double), cudaMemcpyHostToDevice));

	gpu_distance<<<nblock, nthread>>>(d_data, d_distance, d_point, N, DIM);

	checkCudaErrors(
			cudaMemcpy(distance, d_distance, N*sizeof(double), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_distance));
	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_point));
}

int multiple_of_32(int n){
	if(n%32 == 0)
		return n;
	else
		return 32 * (n/32) + 32;
}

void gpu_compute_distance_withreduction(double* data, double* point, double* distance) {
	int datasize = N * DIM * sizeof(double);

	int block_dim_x = multiple_of_32(DIM);
	int block_dim_y = MAX_THREADS_PER_BLOCK / block_dim_x;

	int nblock = N / block_dim_y;
	if(N%block_dim_y != 0)
		nblock++;

	dim3 dim_block(block_dim_x, block_dim_y, 1);

	printf("\nGrid dimension : %d\n", nblock);
	printf("Block dimension : %d * %d\n", dim_block.x, dim_block.y);

	double *d_data;
	double *d_point;
	double *d_distance;

	checkCudaErrors(cudaMalloc((void**)&d_data, datasize));
	checkCudaErrors(cudaMalloc((void**)&d_distance, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_point, DIM*sizeof(double)));

	checkCudaErrors(
			cudaMemcpy(d_data, data, datasize, cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(d_point, point, DIM*sizeof(double), cudaMemcpyHostToDevice));

	gpu_distance_withreduction<<<nblock, dim_block, block_dim_y*block_dim_x*sizeof(double)>>>(d_data, d_distance, d_point, N, DIM);

	checkCudaErrors(
			cudaMemcpy(distance, d_distance, N*sizeof(double), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_distance));
	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_point));
}

/*

int gpu_knn(int * cdata_c, int * data_c, int * point_c, int nclass) {

	int datasize = N * DIM * sizeof(int);

	int nblock = N / blocksize, nthread = blocksize;

	double *distance = new double[N];

	int *d_data;
	int *d_point;
	double *d_distance;

	checkCudaErrors(cudaMalloc((void**)&d_data, datasize));
	checkCudaErrors(cudaMalloc((void**)&d_distance, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_point, DIM*sizeof(int)));

	checkCudaErrors(
			cudaMemcpy(d_data, data_c, datasize, cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(d_point, point_c, DIM*sizeof(int), cudaMemcpyHostToDevice));

	gpu_distance<<<nblock, nthread>>>(d_data, d_distance, d_point, N, DIM);

	checkCudaErrors(
			cudaMemcpy(distance, d_distance, N*sizeof(double), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_distance));
	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_point));

	return -1;
}

*/

