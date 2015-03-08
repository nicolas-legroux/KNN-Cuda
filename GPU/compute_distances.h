/*
 * gpu_knn.h
 *
 *  Created on: Feb 16, 2015
 *      Author: hugo.braun
 */

#ifndef GPU_KNN_H_
#define GPU_KNN_H_

void gpu_compute_distance(double* data, double* point, double* distance);
void gpu_compute_distance_withreduction(double* data, double* point, double* distance);
void gpu_compute_distances(double *train_data, double *test_data, int n_train, int n_test, int dim, double* distances);

#endif /* GPU_KNN_H_ */
