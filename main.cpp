#include "utilities.h"
#include "GPU/compute_distances.h"

int main(){
	int n_train = 3;
	int n_test = 4;
	int dim = 2;

	double *train_data = new double[n_train*dim];
	double *test_data = new double[n_test*dim];
	array_fill(train_data, n_train*dim);
	array_fill(test_data, n_test*dim);
	print_vectors_in_row_major_order(train_data, n_train, dim);
	print_vectors_in_row_major_order(test_data, n_test, dim);

	double *distances = new double[n_train*n_test];

	gpu_compute_distances(train_data, test_data, n_train, n_test, dim, distances);

	array_print(distances, n_train*n_test);

	print_vectors_in_row_major_order(distances, n_test, n_train);


}
