/*
 * gpu_knn.h
 *
 *  Created on: Feb 16, 2015
 *      Author: hugo.braun
 */

#ifndef GPU_KNN_H_
#define GPU_KNN_H_

int gpu_knn(int * cdata_c, int * data_c, int * point_c, int nclass);
void gpu_compute_distance(double* data, double* point, double* distance);
void gpu_compute_distance_withreduction(double* data, double* point, double* distance);


#endif /* GPU_KNN_H_ */
