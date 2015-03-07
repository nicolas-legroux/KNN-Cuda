/*
 * utilities.cpp
 *
 *  Created on: Mar 7, 2015
 *      Author: nicolas.legroux
 */

#include <iostream>
#include <fstream>

#include <sstream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void print_elapsed(clock_t start, clock_t stop)
{
	double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
	printf("Elapsed time: %.3fs\n", elapsed);
}

double random_double() {
	return (double) rand() / (double) RAND_MAX;
}

void array_print(double *arr, int length) {
	int i;
	for (i = 0; i < length; ++i) {
		printf("%1.5f ", arr[i]);
	}
	printf("\n");
}

void array_print_sqrt(double *arr, int length) {
	int i;
	for (i = 0; i < length; ++i) {
		printf("%1.5f ", sqrt(arr[i]));
	}
	printf("\n");
}

void array_fill(double *arr, int length) {
	int i;
	for (i = 0; i < length; ++i) {
		arr[i] = random_double();
	}
}


