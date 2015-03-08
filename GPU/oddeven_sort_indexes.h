/*
 * oddeven_sort.h
 *
 *  Created on: Mar 7, 2015
 *      Author: nicolas.legroux
 */

#ifndef ODDEVEN_SORT_H_
#define ODDEVEN_SORT_H_

void oddeven_sort_indexes(double *data, int * indexes, int n);
void oddeven_sort_indexes_multiple(double *distances, int * indexes, int n_train, int n_test);
void test_oddeven_sort(int DATA_SIZE);
void test_oddeven_sort_multiple(int n_train, int n_test);


#endif /* ODDEVEN_SORT_H_ */
