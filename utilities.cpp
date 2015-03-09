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

int random_boolean(){
	if(random_double()>0.5)
		return 0;
	else
		return 1;
}

void array_print(double *arr, int length) {
	int i;
	for (i = 0; i < length; ++i) {
		printf("%1.5f ", arr[i]);
	}
	printf("\n");
}

void array_fill_boolean(int * arr, int length){
	for(int i=0; i < length; i++){
		arr[i] = random_boolean();
	}
}

void array_print(int *arr, int length) {
	int i;
	for (i = 0; i < length; ++i) {
		printf("%d ", arr[i]);
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

void array_copy(double *original, double* copy, int length){
	for(int i=0; i<length; i++){
		copy[i] = original[i];
	}
}

void print_vectors_in_column_major_order(double *data, int n, int dim){

	printf("\nThere are %d vectors, each of which is of dimension %d.\n", n, dim);

	for(int i=0; i<n; i++){

		printf("\nNow printing vector #%d \n[ ", (i+1));

		for(int j=0; j<dim; j++){
			printf("%1.5f ",data[j*n+i]);
		}

		printf("]\n");
	}
}

void print_vectors_in_row_major_order(double *data, int n, int dim){

	printf("\nThere are %d vectors, each of which is of dimension %d.\n", n, dim);

	for(int i=0; i<n; i++){

		printf("\nNow printing vector #%d \n[ ", (i+1));

		for(int j=0; j<dim; j++){
			printf("%1.5f ",data[i*dim+j]);
		}

		printf("]\n");
	}
}

void print_vectors_in_row_major_order(int *data, int n, int dim){

	printf("\nThere are %d vectors, each of which is of dimension %d.\n", n, dim);

	for(int i=0; i<n; i++){

		printf("\nNow printing vector #%d \n[ ", (i+1));

		for(int j=0; j<dim; j++){
			printf("%d ",data[i*dim+j]);
		}

		printf("]\n");
	}
}

void convert_row_major_to_column_major(double *data, int n, int dim){
	double * copy = new double[n*dim];

	for(int i=0; i<n*dim; i++){
		copy[i] = data[i];
	}

	for(int i=0; i<dim; i++){
		for(int j=0; j<n; j++){
			data[ i*n + j] = copy[j*dim+i];
		}
	}

	delete[] copy;
}

int multiple_of_32(int n) {
	if (n % 32 == 0)
		return n;
	else
		return 32 * (n / 32) + 32;
}

