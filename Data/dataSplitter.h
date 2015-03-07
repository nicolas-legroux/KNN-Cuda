/*
 * dataSplitter.h
 *
 *  Created on: Mar 7, 2015
 *      Author: hugo.braun
 */

#ifndef DATASPLITTER_H_
#define DATASPLITTER_H_

#include <string>

using namespace std;

void splitData(int * data, int * traindata, int * testdata, int * cdata,
		int * traincdata, int* testcdata, int n, int dim, int from, int to);

void getPoint(int * data, int * point, int i, int dim);


#endif /* DATASPLITTER_H_ */
