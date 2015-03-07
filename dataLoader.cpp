#include <iostream>
#include <fstream>

#include <sstream>
#include "dataLoader.h"
#include <math.h>
#include "configuration.h"

#include "CPU/cpu_knn.h"
#include "GPU/gpu_knn.h"

#include "dataSplitter.h"


using namespace std;
/*

int main() {
	/*
	int * data = new int[DIM*N];
	int * cdata = new int[N];

	loadData("data.csv", cdata, data);

	int ntest = N / 10;

	for(int i = 0; i < 9; i++) {
		int * cdatatest = new int[ntest];
		int * cdatatrain = new int[N - ntest];

		int * datatest = new int[ntest*DIM];
		int * datatrain = new int[(N - ntest)*DIM];

		splitData(data, datatrain, datatest, cdata, cdatatrain, cdatatest, N, DIM, i * ntest, (i+1)*ntest);


		int itest = 0;
		int fails = 0;

		for(int t = 0; t < ntest; t++) {
			int * point = new int[DIM];

			getPoint(datatest, point, t, DIM);





			if(gpu_knn(cdatatrain, data, point, 125) != cdatatest[t])
				fails++;
			itest++;
		}

		double r = (double)fails / (double)itest;

		cout << "Error rate on test " << i << " : " << r << endl;


	}



	return 0;

}
*/

void loadData(string filename, int * cdata, int * data) {
	ifstream file;
	string line;
	string cell;

	file.open (filename.c_str());
		int i = 0;

		if (file.is_open()) {
			while (getline (file,line) ) {
			int j = 0;
			stringstream  lineStream(line);
			getline(lineStream,cell,',');
			int c = atoi(cell.c_str());
			cdata[i] = c;

			while(getline(lineStream,cell,',')) {
				int d = atoi(cell.c_str());

				data[DIM * i + j] = d;
				j++;
			}

			i++;
			}
			file.close();
		} else {
			cout << "Could not open file !";
			exit(1);
		}

		cout << "Done loading data" << endl;
}
