/*
 * quicksort_sort_indexes.h
 *
 *  Created on: Mar 8, 2015
 *      Author: hugo.braun
 */

#ifndef QUICKSORT_SORT_INDEXES_H_
#define QUICKSORT_SORT_INDEXES_H_

void gpu_quicksort(double * data, int * indexes, int n, int dim);
void gpu_quicksort_benchmark(double * data, int n, int dim);

#endif /* QUICKSORT_SORT_INDEXES_H_ */
