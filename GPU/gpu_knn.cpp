/*
 * gpu_knn.cpp
 *
 *  Created on: 8 mars 2015
 *      Author: nicolas.legroux
 */

#include "../configuration.h"
#include "compute_distances.h"
#include "oddeven_sort_indexes.h"

int gpu_knn(int * class_data, double * data, double * point, int nclass, int n, int dim, int k){

	double * distance = new double[n];
	gpu_compute_distance(data, point, distance);

	int * indexes = new int[n];
	for(int i=0; i<n; i++){
		indexes[i] = i;
	}

	oddeven_sort_indexes(distance, indexes, n);

	int * class_count = new int[nclass];

	for(int i=0; i<k; i++){
		class_count[class_data[indexes[i]]]++;
	}

	int c = 0;
	int max_c = class_count[0];

	for(int i=0; i<nclass; i++){
		if(class_count[i]>max_c){
			c = i;
			max_c = class_count[i];
		}
	}

	return c;
}


