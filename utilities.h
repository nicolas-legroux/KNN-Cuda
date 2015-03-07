/*
 * utilities.h
 *
 *  Created on: Mar 7, 2015
 *      Author: nicolas.legroux
 */

#ifndef UTILITIES_H_
#define UTILITIES_H_

void print_elapsed(clock_t start, clock_t stop);
double random_double();
void array_print(double *arr, int length);
void array_print_sqrt(double *arr, int length);
void array_fill(double *arr, int length);

#endif /* UTILITIES_H_ */
