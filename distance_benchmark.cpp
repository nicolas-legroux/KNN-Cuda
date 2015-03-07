/*
 * distance_benchmark.cpp
 *
 *  Created on: Mar 7, 2015
 *      Author: nicolas.legroux
 */

#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "configuration.h"
#include "CPU/cpu_knn.h"
#include "GPU/gpu_knn.h"
#include "utilities.h"

void compute_benchmark() {

	clock_t start, stop;

	int ITERATION = 1;

	srand(time(NULL));

	double * data = new double[DIM * N];
	array_fill(data, DIM*N);

	double * point = new double[DIM];
	array_fill(point, DIM);

	//printf("\nDATA ARRAY : \n");
	//array_print(data, DIM * N);

	//printf("\nPOINT ARRAY : \n");
	//array_print(point, DIM);

	double * distance_simple = new double[N];
	double * distance_withreduction = new double[N];

	start = clock();
	for (int i = 0; i < ITERATION; i++) {
		gpu_compute_distance(data, point, distance_simple);
		//printf("Done with pass %d\n", i);
	}
	stop = clock();

	printf("\nBenchmark with no pointer leap :\n");
	print_elapsed(start, stop);

	start = clock();
	for (int i = 0; i < ITERATION; i++) {
		gpu_compute_distance_withreduction(data, point, distance_withreduction);
		//printf("Done with pass %d\n", i);
	}
	stop = clock();

	printf("\nBenchmark with pointer leap :\n");
	print_elapsed(start, stop);

	//printf("\nDISTANCE ARRAY : \n");
	//array_print_sqrt(distance_simple, N);

	//printf("\nDISTANCE WITH REDUCTION ARRAY : \n");
	//array_print_sqrt(distance_withreduction, N);
}

