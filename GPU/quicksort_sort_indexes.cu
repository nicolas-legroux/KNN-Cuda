#include <stdio.h>
#include <stdlib.h>
#include "helper_functions.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#import "../utilities.h"


__device__ void swap(double* data, int* indexes, int i, int j) {
	double t = data[j];
	data[j] = data[i];
	data[i] = t;

	int ti = indexes[j];
	indexes[j] = indexes[i];
	indexes[i] = ti;
}


/*
 * Naive sort
 * used if the quicksort uses too many levels
 */__device__ void naivesort(double *data, int * indexes, int left, int right) {
	for (int i = left; i <= right; i++) {
		double min = data[i];
		int imin = i;
		for (int j = i + 1; j <= right; j++) {
			int vj = data[j];
			if (vj < min) {
				imin = j;
				min = vj;
			}
		}
		if (i != imin)
			swap(data, indexes, i, imin);
	}
}


__global__ void k_quicksort(double* data, int * indexes, int dim, int n) {
	#define STACKSIZE 20

	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idy >= n)
		return;

	int L, R;
	double pivot;

	int left[STACKSIZE];
	int right[STACKSIZE];

	int istack = 0;

	left[istack] = idy * dim;
	right[istack] = idy * dim + dim - 1;

	while (istack >= 0) {
		L = left[istack];
		R = right[istack];

		if (L < R) {
			pivot = data[L];
			int pos = L;

			swap(data, indexes, L, R);

			for (int i = L; i < R; i++) {

				if (data[i] < pivot) {

					if(i != pos)
						swap(data, indexes, i, pos);
					//printf("swaping %i and %i \n", i, pos);
					pos++;
				}
			}

			if(R != pos)
				swap(data, indexes, pos, R);

			istack--;
			if ((istack + 1) > STACKSIZE) {
				if (pos - 1 > L)
					naivesort(data, indexes, L, pos - 1);

				if (pos + 1 < R)
					naivesort(data, indexes, pos + 1, R);
			} else {
				if (pos - 1 > L) {
					istack++;
					left[istack] = L;
					right[istack] = pos - 1;
				}
				if (pos + 1 < R) {
					istack++;
					left[istack] = pos + 1;
					right[istack] = R;

				}

			}
		} else
			istack--;

	}
}

void gpu_quicksort(double * data, int n, int dim) {

	int * indexes = new int[n*dim];
	for (int i = 0; i < n; i++)
		for(int j = 0; j < dim; j++)
			indexes[i * dim + j] = j;

	int datasize_double = dim * n * sizeof(double);
	int datasize_int = dim * n * sizeof(int);

	double *d_data;
	int * d_indexes;

	checkCudaErrors(cudaMalloc((void**) &d_data, datasize_double));
	checkCudaErrors(cudaMalloc((void**) &d_indexes, datasize_int));

	checkCudaErrors(
			cudaMemcpy(d_data, data, datasize_double, cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(d_indexes, indexes, datasize_int, cudaMemcpyHostToDevice));

	int nthreads = 1024;
	dim3 dim_grid(1, n / nthreads + 1);
	dim3 dim_block(1, nthreads);

	printf("Nblock : %i \n", n / nthreads + 1);

	k_quicksort<<<dim_grid, dim_block>>>(d_data, d_indexes, dim, n);

	checkCudaErrors(
			cudaMemcpy(data, d_data, datasize_double, cudaMemcpyDeviceToHost));



	checkCudaErrors(
			cudaMemcpy(indexes, d_indexes, datasize_int, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_data));
	checkCudaErrors(cudaFree(d_indexes));

	print_vectors_in_row_major_order(indexes, n, dim);
	printf("hello");
}

void gpu_quicksort_benchmark(double * data, int n, int dim) {
	printf("------------------------------------------\n");
	printf("Starting benchmark for gpu quicksort (array size = %i)\n", n);
	printf("------------------------------------------\n");

	print_vectors_in_row_major_order(data, n, dim);
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	gpu_quicksort(data, n, dim);

	sdkStopTimer(&timer);

	print_vectors_in_row_major_order(data, n, dim);

	printf("------------------------------------------\n");
	printf("Sorting of array size %i done. Processing time on GPU : %f (ms)\n",
			n * dim, sdkGetTimerValue(&timer));
	printf("------------------------------------------\n");
	printf("\nTesting results...\n");

	for (int i = 0; i < n; i++) {
		for (int x = 0; x < dim-1; x++) {
			if (data[i * dim + x] > data[i * dim + x + 1]) {
				printf("Sorting failed for %i \n", x);
				break;
			}
		}
	}
}
