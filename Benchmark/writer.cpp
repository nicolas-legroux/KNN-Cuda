/*
 * writer.cpp
 *
 *  Created on: Mar 9, 2015
 *      Author: nicolas.legroux
 */

#include <string>
#include <iostream>
#include <fstream>

using namespace std;

void writer(string filename, int* parameter, double* result1, double*result2,
		int n) {
	ofstream myfile(filename.c_str());
	if (myfile.is_open()) {
		for(int i=0; i<n; i++){
			myfile << parameter[i] << "\t" << result1[i] << "\t" << result2[i] << endl;
		}
		myfile.close();
		cout << "Done writing to file" << endl;
	} else
		cout << "Unable to open file";
}

