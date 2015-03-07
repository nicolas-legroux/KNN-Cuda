/*
 * knn.cpp
 *
 *  Created on: Feb 9, 2015
 *      Author: hugo.braun
 */

#import "cpu_knn.h"

#include <iostream>

using namespace std;


#import <math.h>


int cpu_knn(int * cdata_c, int * data_c, int * point_c, int nclass) {


	int cdata[NLEARN];
	int data[NLEARN][DIM];

	int knn[K][DIM];
	int cknn[K];

	int point[DIM];

	//Converting

	for(int j = 0; j < DIM; j++)
		point[j] = point_c[j];

	for(int i = 0; i < NLEARN; i++)
		for(int j = 0; j < DIM; j++)
			data[i][j] = data_c[i * DIM  + j];


	for(int i = 0; i < NLEARN; i++)
			cdata[i] = cdata_c[i];

	//Done converting
	for(int i = 0; i < K; i++) {
		for(int j = 0; j < DIM; j++)
			knn[i][j] = data[i][j];
		cknn[i] = cdata[i];
	}

	float worstDistance = findWorstDistance(point, knn);

	for(int j = 0; j < NLEARN; j++) {
		if(distance(data[j],point) < worstDistance) {
			injectPoint(point, data[j], knn, cknn, cdata[j]);
		}
	}

	int pointperclass[nclass];

	for(int i = 0; i < nclass; i++)
		pointperclass[i] = 0;

	for(int i =0; i < K; i++)
		pointperclass[cknn[i]]++;

	int cmax = 0;
	int c = 0;

	for(int i = 0; i < nclass; i++)
		if(pointperclass[i] > cmax) {
			cmax = pointperclass[i];
			c = i;
		}

	return c;
}

void injectPoint(const int (&point)[DIM], const int (&newNeighbor)[DIM], int (&knn)[K][DIM], int (&cknn)[K], int c) {
	float worst = 0;
	int iworst;

	for(int i = 0; i < K; i++)
		if(worst < distance(point, knn[i])) {
			worst = distance(point, knn[i]);
			iworst = i;
		}

	for(int j = 0; j < DIM; j++)
		knn[iworst][j] = newNeighbor[j];

	cknn[iworst] = c;
}

float findWorstDistance(const int (&point)[DIM], const int (&knn)[K][DIM]) {
	float worst = 0;
	for(int i = 0; i < K; i++)
		if(worst < distance(point, knn[i]))
			worst = distance(point, knn[i]);

	return worst;
}

float distance(const int (&p1)[DIM], const int (&p2)[DIM]) {
	int d = 0;

	for(int i = 0; i<DIM; i++)
		d += abs(p1[i] - p2[i]);

	return d;
}

