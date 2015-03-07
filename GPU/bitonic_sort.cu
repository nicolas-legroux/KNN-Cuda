/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc -arch=sm_11 bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "../utilities.h"

/* Every thread gets exactly one value in the unsorted array. */
#define THREADS 512 // 2^9
#define BLOCKS 32768 // 2^15
#define NUM_VALS THREADS*BLOCKS


__global__ void bitonic_sort_step(double *dev_values, int j, int k) {
	unsigned int i, ixj; /* Sorting partners: i and ixj */
	i = threadIdx.x + blockDim.x * blockIdx.x;
	ixj = i ^ j;

	/* The threads with the lowest ids sort the array. */
	if ((ixj) > i) {
		if ((i & k) == 0) {
			/* Sort ascending */
			if (dev_values[i] > dev_values[ixj]) {
				/* exchange(i,ixj); */
				double temp = dev_values[i];
				dev_values[i] = dev_values[ixj];
				dev_values[ixj] = temp;
			}
		}
		if ((i & k) != 0) {
			/* Sort descending */
			if (dev_values[i] < dev_values[ixj]) {
				/* exchange(i,ixj); */
				double temp = dev_values[i];
				dev_values[i] = dev_values[ixj];
				dev_values[ixj] = temp;
			}
		}
	}
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(double *values) {
	double *dev_values;
	size_t size = NUM_VALS * sizeof(double);

	cudaMalloc((void**) &dev_values, size);
	cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

	dim3 blocks(BLOCKS, 1); /* Number of blocks */
	dim3 threads(THREADS, 1); /* Number of threads */

	int j, k;
	/* Major step */
	for (k = 2; k <= NUM_VALS; k <<= 1) {
		/* Minor step */
		for (j = k >> 1; j > 0; j = j >> 1) {
			bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
		}
	}
	cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
	cudaFree(dev_values);
}

void test_bitonic_sort() {
	clock_t start, stop;

	double *values = (double*) malloc(NUM_VALS * sizeof(double));
	array_fill(values, NUM_VALS);

	start = clock();
	bitonic_sort(values); /* Inplace */
	stop = clock();

	print_elapsed(start, stop);

	printf("\nTesting results...\n");
	for (int x = 0; x < NUM_VALS - 1; x++) {
		if (values[x] > values[x + 1]) {
			printf("Sorting failed.\n");
			break;
		} else if (x == NUM_VALS - 2)
			printf("SORTING SUCCESSFUL\n");
	}
}
