#include "utilities.h"
#include "GPU/compute_distances.h"
#include "GPU/oddeven_sort_indexes.h"
#include "Data/data_loader.h"
#include "CPU/cpu_knn.h"
#include "GPU/gpu_knn.h"
#include "GPU/gpu_average.h"
#include "GPU/extract_minimums.h"
#include "GPU/compute_distances.h"
#include "GPU/quicksort_sort_indexes.h"
#include "Benchmark/benchmark.h"

void test_distance(int n_train, int n_test, int dim) {
	test_distances_multiple(n_train, n_test, dim);
}

void test1(){
	int k = 5;
	int n_train = 100;
	int n_labels = 2;
	int n_test = 20;
	int dim = 150;
	double * train_data = new double[n_train * dim];
	int * train_labels = new int[n_train];
	double * test_data = new double[n_test * dim];
	array_fill(train_data, n_train * dim);
	array_fill(test_data, n_test * dim);

	array_fill_boolean(train_labels, n_train);
	int * real_test_labels = new int[n_test];
	int * knn_test_labels = new int[n_test];
	//loadData("data.csv", train_labels, train_data, dim);
	//loadData("test.csv", real_test_labels, test_data, dim);
	cpu_knn_benchmark(train_data, test_data, train_labels, n_train, n_test,
			n_labels, dim, k, knn_test_labels);
	//array_print(real_test_labels, n_test);
	array_print(knn_test_labels, n_test);
	gpu_knn_extract_minimums_benchmark(train_data, test_data, train_labels, n_train,
			n_test, n_labels, dim, k, knn_test_labels);
	array_print(knn_test_labels, n_test);
}

void test2(){
	int k = 5;
	int n_train = 50000;
	int n_labels = 2;
	int n_test = 2000;
	int dim = 150;
	double * train_data = new double[n_train * dim];
	int * train_labels = new int[n_train];
	double * test_data = new double[n_test * dim];
	array_fill(train_data, n_train * dim);
	array_fill(test_data, n_test * dim);

	array_fill_boolean(train_labels, n_train);
	int * real_test_labels = new int[n_test];
	int * knn_test_labels = new int[n_test];

	//loadData("data.csv", train_labels, train_data, dim);
	//loadData("test.csv", real_test_labels, test_data, dim);
	cpu_knn_benchmark(train_data, test_data, train_labels, n_train, n_test,
			n_labels, dim, k, knn_test_labels);
	//array_print(real_test_labels, n_test);
	//array_print(knn_test_labels, n_test);
	gpu_knn_extract_minimums_benchmark(train_data, test_data, train_labels, n_train,
			n_test, n_labels, dim, k, knn_test_labels);
	//array_print(knn_test_labels, n_test);
}

int main() {

	//benchmark_OE_n_train();
	//benchmark_EM_dim();
	//test1();
	test2();

	printf("\nFinished without error.");
}
