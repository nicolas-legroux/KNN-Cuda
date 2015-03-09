/*
 * extract_minimum.h
 *
 *  Created on: Mar 8, 2015
 *      Author: nicolas.legroux
 */

#ifndef EXTRACT_MINIMUM_H_
#define EXTRACT_MINIMUM_H_

void find_k_minimums(double * data, int n_train, int n_test, int k, int * k_minimum_indexes);
void test_extract_minimum(int n_train, int n_test, int k);
int find_minimum_cpu(double * data, int * indexes, int n);



#endif /* EXTRACT_MINIMUM_H_ */
