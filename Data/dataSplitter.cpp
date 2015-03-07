#include "dataSplitter.h"

void splitData(int * data, int * traindata, int * testdata, int * cdata,
		int * traincdata, int* testcdata, int n, int dim, int from, int to) {

	int itest = 0;
	int itrain = 0;

	for(int ipoint = 0; ipoint < n; ipoint++) {

		if(ipoint >= from && ipoint <= to) {
			for(int i = 0; i < dim; i++)
				testdata[itest * dim + i] = data[ipoint * dim + i];

			traincdata[itest] = cdata[ipoint];
			itest++;
		} else {
			for(int i = 0; i < dim; i++)
				traindata[itrain * dim + i] = data[ipoint * dim + i];

			testcdata[itest] = cdata[ipoint];
			itrain++;
		}
	}

}

void getPoint(int * data, int * point, int in, int dim) {
	for(int i = 0; i < dim; i++)
		point[i] = data[in * dim + i];
}
