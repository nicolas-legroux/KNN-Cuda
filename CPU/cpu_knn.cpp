/*
 * knn.cpp
 *
 *  Created on: Feb 9, 2015
 *      Author: hugo.braun
 */

#import "cpu_knn.h"
#include <limits>

#include <iostream>
#import "../utilities.h"

using namespace std;


#import <math.h>


void cpu_knn(double * train_data, double * test_data, int * train_labels,
		int n_train, int n_test, int n_labels, int dim, int k, int * knn_labels) {

	for(int i = 0; i < n_test; i++) {
		double * lsdistance = new double[n_train];

		for(int j = 0; j < n_train; j++) {
			lsdistance[j] = distance(train_data, test_data, j, i, dim);
		}

		int * labelsCount = new int[n_labels];
		for(int l = 0; l < n_labels; l++)
			labelsCount[l] = 0;

		for(int ik = 0; ik < k; ik++) {
			int idxmin = find_index_min_naive(lsdistance, n_train);
			labelsCount[train_labels[idxmin]]++;
			lsdistance[idxmin] = numeric_limits<double>::max();
		}

		knn_labels[i] = find_index_max_naive(labelsCount, n_labels);

		delete[] labelsCount;
		delete[] lsdistance;
	}
}

void cpu_knn_benchmark(double * train_data, double * test_data, int * train_labels,
		int n_train, int n_test, int n_labels, int dim, int k, int * knn_labels) {
	clock_t start = clock();
	cpu_knn(train_data, test_data, train_labels, n_train, n_test, n_labels, dim, k, knn_labels);
	clock_t stop = clock();

	printf("\nResult of benchmark for CPU : \n");
	print_elapsed(start, stop);
}

//Compute distance between point p1 and the point at index idx of the 2D array data
double distance(double * matrix1, double * matrix2, int idx1, int idx2, int dim) {
	double d = 0.0;

	for(int i = 0; i<dim; i++){
		double temp = matrix1[idx1 * dim + i] - matrix2[idx2 * dim + i];
		d += temp*temp;
	}

	return d;
}

int find_index_min_naive(double * vector, int dim) {
	int imin = 0;
	double min = vector[0];
	for(int i = 1; i < dim; i++) {
		if(min > vector[i]) {
			imin = i;
			min = vector[i];
		}
	}

	return imin;
}

int find_index_max_naive(int * vector, int dim) {
	int imax = 0;
	int max = vector[0];
	for(int i = 1; i < dim; i++) {
		if(max < vector[i]) {
			imax = i;
			max = vector[i];
		}
	}

	return imax;
}
