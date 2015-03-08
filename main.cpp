#include "utilities.h"
#include "GPU/compute_distances.h"
#include "GPU/oddeven_sort_indexes.h"

int main(){

	// TESTING MULTIPLE DISTANCE CALCULATION
	test_distances_multiple();

	//TESTING MULTIPLE SORTING
	test_oddeven_sort_multiple();

}
