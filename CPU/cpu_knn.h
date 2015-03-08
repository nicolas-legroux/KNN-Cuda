/*
 * cpu_knn.h
 *
 *  Created on: Feb 9, 2015
 *      Author: hugo.braun
 */

#ifndef CPU_KNN_H_
#define CPU_KNN_H_

#import "../configuration.h"


double distance(double * matrix1, double * matrix2, int idx1, int idx2, int dim);

void cpu_knn(double * train_data, double * test_data, int * train_labels,
		int n_train, int n_test, int n_labels, int dim, int k, int * knn_labels);
void cpu_knn_benchmark(double * train_data, double * test_data, int * train_labels,
		int n_train, int n_test, int n_labels, int dim, int k, int * knn_labels);

int find_index_min_naive(double * vector, int dim);
int find_index_max_naive(int * vector, int dim);
#endif /* CPU_KNN_H_ */
