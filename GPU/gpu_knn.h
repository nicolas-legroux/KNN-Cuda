/*
 * gpu_knn.h
 *
 *  Created on: 8 mars 2015
 *      Author: nicolas.legroux
 */

#ifndef GPU_KNN_H_
#define GPU_KNN_H_

void gpu_knn(double *train_data, double *test_data, int *train_labels, int n_train, int n_test,
		int n_labels, int dim, int k, int* knn_labels);
void gpu_knn_benchmark(double *train_data, double *test_data, int *train_labels, int n_train, int n_test,
		int n_labels, int dim, int k, int* knn_labels);

#endif /* GPU_KNN_H_ */
