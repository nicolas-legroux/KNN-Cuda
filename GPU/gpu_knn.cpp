/*
 * gpu_knn.cpp
 *
 *  Created on: 8 mars 2015
 *      Author: nicolas.legroux
 */

#include <ctime>
#include "../configuration.h"
#include "../utilities.h"
#include "compute_distances.h"
#include "oddeven_sort_indexes.h"
#include "extract_minimums.h"
#include "quicksort_sort_indexes.h"

//train_data and test_data are assumed to be stored in row-major order
void gpu_knn_oddevensort(double *train_data, double *test_data, int *train_labels,
		int n_train, int n_test, int n_labels, int dim, int k,
		int* knn_labels) {

	double * distances = new double[n_train * n_test];

	gpu_compute_distances(train_data, test_data, n_train, n_test, dim,
			distances);

	int * indexes = new int[n_train * n_test];
	for (int i = 0; i < n_test; i++) {
		for (int j = 0; j < n_train; j++) {
			indexes[i * n_train + j] = j;
		}
	}

	int * class_count = new int[n_labels];

	gpu_quicksort(distances, indexes, n_test, n_train);

	for (int i = 0; i < n_test; i++) {
		//Reset class count
		for (int j = 0; j < n_labels; j++) {
			class_count[j] = 0;
		}

		//Find the class of the k nearest neighbours
		for (int j = 0; j < k; j++) {
			class_count[train_labels[indexes[i * n_train + j]]]++;
		}

		//Find the max class
		int max_class = 0;
		int max = class_count[0];
		for (int j = 0; j < n_labels; j++) {
			if (class_count[j] > max) {
				max_class = j;
				max = class_count[j];
			}
		}

		knn_labels[i] = max_class;
	}

	delete[] distances;
	delete[] class_count;
}

void gpu_knn_quicksort(double *train_data, double *test_data, int *train_labels,
		int n_train, int n_test, int n_labels, int dim, int k,
		int* knn_labels) {

	double * distances = new double[n_train * n_test];

	gpu_compute_distances(train_data, test_data, n_train, n_test, dim,
			distances);

	int * indexes = new int[n_train * n_test];
	for (int i = 0; i < n_test; i++) {
		for (int j = 0; j < n_train; j++) {
			indexes[i * n_train + j] = j;
		}
	}

	int * class_count = new int[n_labels];

	gpu_quicksort(distances, indexes, n_test, n_train);

	for (int i = 0; i < n_test; i++) {
		//Reset class count
		for (int j = 0; j < n_labels; j++) {
			class_count[j] = 0;
		}

		//Find the class of the k nearest neighbours
		for (int j = 0; j < k; j++) {
			class_count[train_labels[indexes[i * n_train + j]]]++;
		}

		//Find the max class
		int max_class = 0;
		int max = class_count[0];
		for (int j = 0; j < n_labels; j++) {
			if (class_count[j] > max) {
				max_class = j;
				max = class_count[j];
			}
		}

		knn_labels[i] = max_class;
	}

	delete[] distances;
	delete[] class_count;
}

void gpu_knn_extract_minimums(double *train_data, double *test_data, int *train_labels,
		int n_train, int n_test, int n_labels, int dim, int k,
		int* knn_labels) {

	double * distances = new double[n_train * n_test];

	gpu_compute_distances(train_data, test_data, n_train, n_test, dim,
			distances);

	int * indexes = new int[n_train * n_test];
	for (int i = 0; i < n_test; i++) {
		for (int j = 0; j < n_train; j++) {
			indexes[i * n_train + j] = j;
		}
	}

	int * class_count = new int[n_labels];
	int * k_minimum_indexes = (int*) malloc(k * n_test * sizeof(int));

	find_k_minimums(distances, n_train, n_test, k, k_minimum_indexes);

	for (int i = 0; i < n_test; i++) {
		//Reset class count
		for (int j = 0; j < n_labels; j++) {
			class_count[j] = 0;
		}

		//Find the class of the k nearest neighbours
		for (int j = 0; j < k; j++) {
			class_count[train_labels[k_minimum_indexes[i * k + j]]]++;
		}

		//Find the max class
		int max_class = 0;
		int max = class_count[0];
		for (int j = 0; j < n_labels; j++) {
			if (class_count[j] > max) {
				max_class = j;
				max = class_count[j];
			}
		}

		knn_labels[i] = max_class;
	}

	delete[] distances;
	delete[] class_count;
	delete[] k_minimum_indexes;
}

double gpu_knn_quicksort_benchmark(double *train_data, double *test_data, int *train_labels,
		int n_train, int n_test, int n_labels, int dim, int k, int* knn_labels) {
	clock_t start = clock();
	gpu_knn_quicksort(train_data, test_data, train_labels, n_train, n_test, n_labels, dim,
			k, knn_labels);
	clock_t stop = clock();
	printf("\nResult of benchmark for GPU KNN with Quicksort :\n");
	double elapsed = print_elapsed(start, stop);
	return elapsed;
}

double gpu_knn_oddevensort_benchmark(double *train_data, double *test_data, int *train_labels,
		int n_train, int n_test, int n_labels, int dim, int k, int* knn_labels) {
	clock_t start = clock();
	gpu_knn_oddevensort(train_data, test_data, train_labels, n_train, n_test, n_labels, dim,
			k, knn_labels);
	clock_t stop = clock();
	printf("\nResult of benchmark for GPU KNN with OddEven sort :\n");
	double elapsed = print_elapsed(start, stop);
	return elapsed;
}


double gpu_knn_extract_minimums_benchmark(double *train_data, double *test_data, int *train_labels,
		int n_train, int n_test, int n_labels, int dim, int k, int* knn_labels) {
	clock_t start = clock();
	gpu_knn_extract_minimums(train_data, test_data, train_labels, n_train, n_test, n_labels, dim,
			k, knn_labels);
	clock_t stop = clock();
	printf("\nResult of benchmark for GPU KNN with Extraction of minimums :\n");
	print_elapsed(start, stop);
	double elapsed = print_elapsed(start, stop);
	return elapsed;
}

