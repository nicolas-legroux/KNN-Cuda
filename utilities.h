/*
 * utilities.h
 *
 *  Created on: Mar 7, 2015
 *      Author: nicolas.legroux
 */

#include <iostream>
#include <fstream>
#include "time.h"
#include <ctime>

#include <sstream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>


#ifndef UTILITIES_H_
#define UTILITIES_H_

void print_elapsed(clock_t start, clock_t stop);
double random_double();
void array_print(double *arr, int length);
void array_print(int * arr, int length);
void array_print_sqrt(double *arr, int length);
void array_fill(double *arr, int length);
void array_fill_boolean(int * arr, int length);
void array_copy(double *original, double* copy, int length);
void print_vectors_in_column_major_order(double *data, int n, int dim);
void print_vectors_in_row_major_order(double *data, int n, int dim);
void print_vectors_in_row_major_order(int *data, int n, int dim);
void convert_row_major_to_column_major(double *data, int n, int dim);
int multiple_of_32(int n);


#endif /* UTILITIES_H_ */
