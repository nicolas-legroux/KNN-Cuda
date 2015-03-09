/*
 * benchmark_quicksort.cpp
 *
 *  Created on: Mar 9, 2015
 *      Author: nicolas.legroux
 */

#include <string>
#include "writer.h"
#include "../utilities.h"
#include "../CPU/cpu_knn.h"
#include "../GPU/gpu_knn.h"

using namespace std;

void benchmark_quicksort_k() {

	int k_min = 1;
	int k_max = 100;
	int k_step = 2;

	string filename = "benchmark.txt";
	int n = (k_max - k_min) / k_step + 1;
	int* parameter = (int*) malloc(n * sizeof(double));
	double* result1 = (double*) malloc(n * sizeof(double));
	double* result2 = (double*) malloc(n * sizeof(double));

	int n_train = 500;
	int n_labels = 2;
	int n_test = 10000;
	int dim = 100;
	double * train_data = new double[n_train * dim];
	int * train_labels = new int[n_train];
	double * test_data = new double[n_test * dim];
	array_fill(train_data, n_train * dim);
	array_fill(test_data, n_test * dim);

	array_fill_boolean(train_labels, n_train);
	int * knn_test_labels = new int[n_test];

	int i = 0;
	for (int k = k_min; k <= k_max; k += k_step) {
		cout << endl << "Benchmarking for k=" << k << endl;
		parameter[i] = k;
		result1[i] = cpu_knn_benchmark(train_data, test_data, train_labels,
				n_train, n_test, n_labels, dim, k, knn_test_labels);
		result2[i] = gpu_knn_quicksort_benchmark(train_data, test_data,
				train_labels, n_train, n_test, n_labels, dim, k,
				knn_test_labels);
		i++;
	}

	writer(filename, parameter, result1, result2, n);
}

void benchmark_quicksort_n_test() {

	int n_test_min = 1000;
	int n_test_max = 40000;
	int n_test_step = 2000;

	string filename = "benchmark.txt";
	int n = (n_test_max - n_test_min) / n_test_step + 1;
	int* parameter = (int*) malloc(n * sizeof(double));
	double* result1 = (double*) malloc(n * sizeof(double));
	double* result2 = (double*) malloc(n * sizeof(double));

	int i = 0;
	for (int n_test = n_test_min; n_test <= n_test_max; n_test += n_test_step) {
		cout << endl << "Benchmarking for n_test=" << n_test << endl;

		int n_train = 500;
		int k = 25;
		int n_labels = 2;
		int dim = 50;
		double * train_data = new double[n_train * dim];
		int * train_labels = new int[n_train];
		double * test_data = new double[n_test * dim];
		array_fill(train_data, n_train * dim);
		array_fill(test_data, n_test * dim);

		array_fill_boolean(train_labels, n_train);
		int * knn_test_labels = new int[n_test];

		parameter[i] = n_test;
		result1[i] = cpu_knn_benchmark(train_data, test_data, train_labels,
				n_train, n_test, n_labels, dim, k, knn_test_labels);
		result2[i] = gpu_knn_quicksort_benchmark(train_data, test_data,
				train_labels, n_train, n_test, n_labels, dim, k,
				knn_test_labels);
		i++;
		delete[] train_data;
		delete[] test_data;
		delete[] knn_test_labels;
	}

	writer(filename, parameter, result1, result2, n);
}

void benchmark_EM_k() {

	int k_min = 1;
	int k_max = 100;
	int k_step = 2;

	string filename = "benchmark.txt";
	int n = (k_max - k_min) / k_step + 1;
	int* parameter = (int*) malloc(n * sizeof(double));
	double* result1 = (double*) malloc(n * sizeof(double));
	double* result2 = (double*) malloc(n * sizeof(double));

	int n_train = 500;
	int n_labels = 2;
	int n_test = 10000;
	int dim = 100;
	double * train_data = new double[n_train * dim];
	int * train_labels = new int[n_train];
	double * test_data = new double[n_test * dim];
	array_fill(train_data, n_train * dim);
	array_fill(test_data, n_test * dim);

	array_fill_boolean(train_labels, n_train);
	int * knn_test_labels = new int[n_test];

	int i = 0;
	for (int k = k_min; k <= k_max; k += k_step) {
		cout << endl << "Benchmarking for k=" << k << endl;
		parameter[i] = k;
		result1[i] = cpu_knn_benchmark(train_data, test_data, train_labels,
				n_train, n_test, n_labels, dim, k, knn_test_labels);
		result2[i] = gpu_knn_extract_minimums_benchmark(train_data, test_data,
				train_labels, n_train, n_test, n_labels, dim, k,
				knn_test_labels);
		i++;
	}

	writer(filename, parameter, result1, result2, n);
}

void benchmark_EM_n_test() {

	int n_test_min = 500;
	int n_test_max = 5000;
	int n_test_step = 250;

	string filename = "benchmark.txt";
	int n = (n_test_max - n_test_min) / n_test_step + 1;
	int* parameter = (int*) malloc(n * sizeof(double));
	double* result1 = (double*) malloc(n * sizeof(double));
	double* result2 = (double*) malloc(n * sizeof(double));

	int i = 0;
	for (int n_test = n_test_min; n_test <= n_test_max; n_test += n_test_step) {
		cout << endl << "Benchmarking for n_test=" << n_test << endl;

		int n_train = 4500;
		int k = 10;
		int n_labels = 2;
		int dim = 15;
		double * train_data = new double[n_train * dim];
		int * train_labels = new int[n_train];
		double * test_data = new double[n_test * dim];
		array_fill(train_data, n_train * dim);
		array_fill(test_data, n_test * dim);

		array_fill_boolean(train_labels, n_train);
		int * knn_test_labels = new int[n_test];

		parameter[i] = n_test;
		result1[i] = cpu_knn_benchmark(train_data, test_data, train_labels,
				n_train, n_test, n_labels, dim, k, knn_test_labels);
		result2[i] = gpu_knn_extract_minimums_benchmark(train_data, test_data,
				train_labels, n_train, n_test, n_labels, dim, k,
				knn_test_labels);
		i++;
		delete[] train_data;
		delete[] test_data;
		delete[] knn_test_labels;
	}

	writer(filename, parameter, result1, result2, n);
}

void benchmark_EM_n_train() {

	int n_train_min = 1000;
	int n_train_max = 15000;
	int n_train_step = 500;

	string filename = "benchmark.txt";
	int n = (n_train_max - n_train_min) / n_train_step + 1;
	int* parameter = (int*) malloc(n * sizeof(double));
	double* result1 = (double*) malloc(n * sizeof(double));
	double* result2 = (double*) malloc(n * sizeof(double));

	int i = 0;
	for (int n_train = n_train_min; n_train <= n_train_max; n_train += n_train_step) {
		cout << endl << "Benchmarking for n_test=" << n_train << endl;

		int n_test = 1000;
		int k = 5;
		int n_labels = 2;
		int dim = 50;
		double * train_data = new double[n_train * dim];
		int * train_labels = new int[n_train];
		double * test_data = new double[n_test * dim];
		array_fill(train_data, n_train * dim);
		array_fill(test_data, n_test * dim);

		array_fill_boolean(train_labels, n_train);
		int * knn_test_labels = new int[n_test];

		parameter[i] = n_train;
		result1[i] = cpu_knn_benchmark(train_data, test_data, train_labels,
				n_train, n_test, n_labels, dim, k, knn_test_labels);
		result2[i] = gpu_knn_extract_minimums_benchmark(train_data, test_data,
				train_labels, n_train, n_test, n_labels, dim, k,
				knn_test_labels);
		i++;
		delete[] train_data;
		delete[] test_data;
		delete[] knn_test_labels;
	}

	writer(filename, parameter, result1, result2, n);
}

void benchmark_OE_n_train() {

	int n_train_min = 100;
	int n_train_max = 3000;
	int n_train_step = 100;

	string filename = "benchmark.txt";
	int n = (n_train_max - n_train_min) / n_train_step + 1;
	int* parameter = (int*) malloc(n * sizeof(double));
	double* result1 = (double*) malloc(n * sizeof(double));
	double* result2 = (double*) malloc(n * sizeof(double));

	int i = 0;
	for (int n_train = n_train_min; n_train <= n_train_max; n_train += n_train_step) {
		cout << endl << "Benchmarking for n_test=" << n_train << endl;

		int n_test = 100;
		int k = 25;
		int n_labels = 2;
		int dim = 30;
		double * train_data = new double[n_train * dim];
		int * train_labels = new int[n_train];
		double * test_data = new double[n_test * dim];
		array_fill(train_data, n_train * dim);
		array_fill(test_data, n_test * dim);

		array_fill_boolean(train_labels, n_train);
		int * knn_test_labels = new int[n_test];

		parameter[i] = n_train;
		result1[i] = cpu_knn_benchmark(train_data, test_data, train_labels,
				n_train, n_test, n_labels, dim, k, knn_test_labels);
		result2[i] = gpu_knn_oddevensort_benchmark(train_data, test_data,
				train_labels, n_train, n_test, n_labels, dim, k,
				knn_test_labels);
		i++;
		delete[] train_data;
		delete[] test_data;
		delete[] knn_test_labels;
	}

	writer(filename, parameter, result1, result2, n);
}

void benchmark_EM_dim() {

	int dim_min = 10;
	int dim_max = 250;
	int dim_step = 10;

	string filename = "benchmark.txt";
	int n = (dim_max - dim_min) / dim_step + 1;
	int* parameter = (int*) malloc(n * sizeof(double));
	double* result1 = (double*) malloc(n * sizeof(double));
	double* result2 = (double*) malloc(n * sizeof(double));

	int i = 0;
	for (int dim = dim_min; dim <= dim_max; dim += dim_step) {
		cout << endl << "Benchmarking for n_test=" << dim << endl;

		int n_train = 5000;
		int n_test = 1000;
		int k = 5;
		int n_labels = 2;
		double * train_data = new double[n_train * dim];
		int * train_labels = new int[n_train];
		double * test_data = new double[n_test * dim];
		array_fill(train_data, n_train * dim);
		array_fill(test_data, n_test * dim);

		array_fill_boolean(train_labels, n_train);
		int * knn_test_labels = new int[n_test];

		parameter[i] = dim;
		result1[i] = cpu_knn_benchmark(train_data, test_data, train_labels,
				n_train, n_test, n_labels, dim, k, knn_test_labels);
		result2[i] = gpu_knn_extract_minimums_benchmark(train_data, test_data,
				train_labels, n_train, n_test, n_labels, dim, k,
				knn_test_labels);
		i++;
		delete[] train_data;
		delete[] test_data;
		delete[] knn_test_labels;
	}

	writer(filename, parameter, result1, result2, n);
}



