/*
 * gpu_knn.h
 *
 *  Created on: 8 mars 2015
 *      Author: nicolas.legroux
 */

#ifndef GPU_KNN_H_
#define GPU_KNN_H_

int gpu_knn(int * class_data, double * data, double * point, int nclass, int n, int dim, int k);

#endif /* GPU_KNN_H_ */
