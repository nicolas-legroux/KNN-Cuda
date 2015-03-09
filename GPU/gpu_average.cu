#include <stdio.h>
#include <stdlib.h>
#include "helper_functions.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#import "../utilities.h"

using namespace std;

__global__ void block_average(double * data, double *per_block_avg, int dim) {
	extern __shared__ float sdata[];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int offset_block = blockIdx.x * blockDim.x;

	if (i >= dim)
		return;

	sdata[threadIdx.x] = data[i];


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
		per_block_avg[blockIdx.x] = sdata[0] / poids;
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

	checkCudaErrors(cudaMalloc((void**) &d_averages, sizeof(double) * (num_blocks + 1)));

	block_average<<<num_blocks, block_size, block_size * sizeof(double)>>>(d_data, d_averages, dim);

	block_average<<<1, num_blocks, num_blocks * sizeof(double)>>>(d_averages, d_averages + num_blocks,num_blocks);

	double average = 0;
	checkCudaErrors(cudaMemcpy(&average, d_averages + num_blocks, sizeof(double), cudaMemcpyDeviceToHost));

	cout << "Device avg: " << average << endl;

	for (int i = 0; i < n; i++) {
		double s = 0;
		for(int j = 0; j < dim; j++) {
			s += data[i * dim + j];
		}
		printf("Avg : %f \n", s / dim);
	}


	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_averages));

}
