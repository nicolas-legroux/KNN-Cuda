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

//train_data and test_data are assumed to be stored in row-major order
void gpu_knn(double *train_data, double *test_data, int *train_labels, int n_train, int n_test,
		int n_labels, int dim, int k, int* knn_labels){

	double * distances = new double[n_train*n_test];

	clock_t start = clock();
	gpu_compute_distances(train_data, test_data, n_train, n_test, dim, distances);
	clock_t stop = clock();
	printf("Time to compute distances : \n");
	print_elapsed(start, stop);

	/*
	int * indexes = new int[n_train*n_test];
	for(int i=0; i<n_test; i++){
		for(int j=0; j<n_train; j++){
			indexes[i*n_train+j] = j;
		}
	}

	oddeven_sort_indexes_multiple(distances, indexes, n_train, n_test);
	*/

	int * k_minimum_indexes = (int*) malloc(k*n_test*sizeof(int));

	find_k_minimums(distances, n_train, n_test, k, k_minimum_indexes);

	int * class_count = new int[n_labels];

	for(int i=0; i<n_test; i++){
		//Reset class count
		for(int j=0; j< n_labels; j++){
			class_count[j] = 0;
		}

		//Find the class of the k nearest neighbours
		for(int j=0; j<k; j++){
			class_count[train_labels[k_minimum_indexes[i*k+j]]]++;
		}

		//Find the max class
		int max_class=0;
		int max=class_count[0];
		for(int j=0; j<n_labels; j++){
			if(class_count[j]>max){
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

void gpu_knn_benchmark(double *train_data, double *test_data, int *train_labels, int n_train, int n_test,
		int n_labels, int dim, int k, int* knn_labels){
	clock_t start = clock();
	gpu_knn(train_data, test_data, train_labels, n_train, n_test,
	n_labels, dim, k, knn_labels);
	clock_t stop = clock();
	print_elapsed(start, stop);
}


