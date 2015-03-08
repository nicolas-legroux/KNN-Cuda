/*
 * knn.cpp
 *
 *  Created on: Feb 9, 2015
 *      Author: hugo.braun
 */

#import "cpu_knn.h"
#include <limits>

#include <iostream>
#import "utilities.h"

using namespace std;


#import <math.h>


int cpu_knn(double * train_data, double * test_data, double * train_labels,
		int n_train, int n_test, int n_labels, int dim, int k, int * knn_labels) {

	for(int i = 0; i < n_test; i++) {
		double distance[n_train];

		for(int j = 0; j < n_train; j++)
			distance[i] = distance(train_data, test_data, j, i, dim);

		int labelsCount[n_labels];
		for(int i = 0; i < n_labels; i++)
			labelsCount[i] = 0;

		for(int ik = 0; ik < k; ik++) {
			int idxmin = find_index_min_naive(distance, n_train);
			labelsCount[train_labels[idxmin]]++;
			distance[idxmin] = numeric_limits<double>::max();
		}

		knn_labels[i] = find_index_max_naive(labelsCount, n_labels);
	}

}

void cpu_knn_benchmark(double * train_data, double * test_data, double * train_labels,
		int n_train, int n_test, int n_labels, int dim, int k, int * knn_labels) {
	clock_t start = clock();
	cpu_knn(train_data, test_data, train_labels, n_train, n_test, n_labels, dim, k, knn_labels);
	clock_t stop = clock;

	print_elapsed(start, stop);
}

//Compute distance between point p1 and the point at index idx of the 2D array data
double distance(double * matrix1, double * matrix2, int idx1, int idx2, int dim) {
	int d = 0;

	for(int i = 0; i<dim; i++)
		d += abs(matrix1[idx1 * dim + i] - matrix2[idx2 * dim + i]);

	return d;
}

int find_index_min_naive(double &vector[], int dim) {
	int imin = 0;
	int min = vector[0];
	for(int i = 1; i < dim; i++) {
		if(min > vector[i]) {
			imin = i;
			min = vector[i];
		}
	}

	return imin;
}

int find_index_max_naive(int &vector[], int dim) {
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
