#include <iostream>
#include <fstream>

#include <sstream>
#include "data_loader.h"
#include <math.h>
#include "../configuration.h"

#include "../CPU/cpu_knn.h"
#include "../GPU/compute_distances.h"

#include "dataSplitter.h"

using namespace std;

void loadData(string filename, int * cdata, double * data, int dim) {
	ifstream file;
	string line;
	string cell;

	file.open(filename.c_str());
	int i = 0;

	if (file.is_open()) {
		while (getline(file, line)) {
			int j = 0;
			stringstream lineStream(line);
			getline(lineStream, cell, ',');
			int c = atoi(cell.c_str());
			cdata[i] = c;

			while (getline(lineStream, cell, ',')) {
				double d = atof(cell.c_str());

				data[dim * i + j] = d;
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
