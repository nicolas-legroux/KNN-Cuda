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

#include "../configuration.h"
#include "../GPU/compute_distances.h"
#include "../utilities.h"

void compute_benchmark(int n, int dim) {

	clock_t start, stop;

	int ITERATION = 1;

	srand(time(NULL));

	double * data = new double[n*dim];
	array_fill(data, n*dim);

	double * point = new double[dim];
	array_fill(point, dim);

	//printf("\nDATA ARRAY : \n");
	//array_print(data, DIM * N);

	//printf("\nPOINT ARRAY : \n");
	//array_print(point, DIM);

	double * distance_simple = new double[n];
	double * distance_withreduction = new double[n];

	start = clock();
	for (int i = 0; i < ITERATION; i++) {
		gpu_compute_distance(data, point, n, dim, distance_simple);
		//printf("Done with pass %d\n", i);
	}
	stop = clock();

	printf("\nBenchmark with no pointer leap :\n");
	print_elapsed(start, stop);

	start = clock();
	for (int i = 0; i < ITERATION; i++) {
		gpu_compute_distance_withreduction(data, point, n, dim, distance_withreduction);
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

