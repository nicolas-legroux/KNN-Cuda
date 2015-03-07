/*
 * cpu_knn.h
 *
 *  Created on: Feb 9, 2015
 *      Author: hugo.braun
 */

#ifndef CPU_KNN_H_
#define CPU_KNN_H_

#import "../configuration.h"


float findWorstDistance(const int (&point)[DIM], const int (&knn)[K][DIM]);
float distance(const int (&p1)[DIM], const int (&p2)[DIM]);
void injectPoint(const int (&point)[DIM], const int (&newNeighbor)[DIM], int (&knn)[K][DIM], int (&cknn)[K], int c);

int cpu_knn(int * cdata_c, int * data_c, int * point_c, int nclass);

#endif /* CPU_KNN_H_ */
