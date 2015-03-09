#include <stdio.h>
#include <stdlib.h>
#include "helper_functions.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#import "../utilities.h"

using namespace std;

__global__ void block_average(double * data, double *per_block_avg, int dim, int avgdim) {
	extern __shared__ float sdata[];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int offset_block = blockIdx.x * blockDim.x;

	int idy = blockIdx.y * blockDim.y + threadIdx.y;


	if (i >= dim)
		return;

	sdata[threadIdx.x] = data[idy * dim + i];

	__syncthreads();

	for (int offset = blockDim.x / 2; offset > 0; offset = offset/2) {
		if (threadIdx.x < offset && (offset_block + offset + threadIdx.x) < dim)
			sdata[threadIdx.x] += sdata[threadIdx.x + offset];

		__syncthreads();
	}

	int poids = dim - offset_block;

	if(poids > blockDim.x)
		poids = blockDim.x;

	if (threadIdx.x == 0) {
		per_block_avg[avgdim * idy + blockIdx.x] = sdata[0] / poids;
	}
}

void gpu_average(double * data, int n, int dim) {
	int size_double = sizeof(double) * n * dim;
	double *d_data = 0;

	int block_size = 64;
	int num_blocks = (dim / block_size) + ((dim % block_size) ? 1 : 0);

	checkCudaErrors(cudaMalloc((void**) &d_data, size_double));
	checkCudaErrors(cudaMemcpy(d_data, data, size_double, cudaMemcpyHostToDevice));

	printf("Using %i blocks \n", num_blocks);

	double *d_averages = 0;
	double *d_final_averages = 0;


	checkCudaErrors(cudaMalloc((void**) &d_averages, sizeof(double) * (n*num_blocks)));
	checkCudaErrors(cudaMalloc((void**) &d_final_averages, sizeof(double) * n));

	dim3 dim_grid(num_blocks, n);
	dim3 dim_block(block_size, 1);

	block_average<<<dim_grid, dim_block, block_size * sizeof(double)>>>(d_data, d_averages, dim, num_blocks);

	dim3 dim_grid_final(1, n);
	dim3 dim_block_final(num_blocks, 1);

	block_average<<<dim_grid_final, dim_block_final, num_blocks * sizeof(double)>>>(d_averages, d_final_averages, num_blocks, 1);

	double * final_averages = new double[n];
	checkCudaErrors(cudaMemcpy(final_averages, d_final_averages, n*sizeof(double), cudaMemcpyDeviceToHost));

	for (int i = 0; i < n; i++) {
		double s = 0;
		for(int j = 0; j < dim; j++) {
			s += data[i * dim + j];
		}
		printf("Avg : %f device : %f \n", s / dim, final_averages[i]);
	}


	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_averages));
	checkCudaErrors(cudaFree(d_final_averages));

}
