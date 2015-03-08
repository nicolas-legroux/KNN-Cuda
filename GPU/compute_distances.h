/*
 * gpu_knn.h
 *
 *  Created on: Feb 16, 2015
 *      Author: hugo.braun
 */

#ifndef COMPUTE_DISTANCES_H_
#define COMPUTE_DISTANCES_H_

void gpu_compute_distance(double* train_data, double* test_point, int n_train, int dim, double* distance);
void gpu_compute_distance_withreduction(double* train_data, double* test_point, int n_train, int dim,
		double* distance);
void gpu_compute_distances(double *train_data, double *test_data, int n_train,
		int n_test, int dim, double* distances);
void test_distances_multiple();

#endif /* COMPUTE_DISTANCES_H_ */
