#define NUM_THREADS 1024
#define TOLERANCE 0.001
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "cuutils.h"
using namespace std;

float* moveScoreToCUDA(Scores s, int dim1, int dim2, int dim3) {
	float* cuScore;
	int total = dim1*dim2*dim3;
	cudaMallocHost((void**)&cuScore, total * sizeof(float));
	for (int i = 0; i < dim1; i++){
		for (int j = 0; j < dim2; j++) {
			for (int k = 0; k < dim3; k++) {
				cuScore[k*dim1*dim2 + j*dim1 + i] = s[i][j][k];
			}
		}
	}
	float* deviceScore;
	cudaMalloc((void**)&deviceScore, total * sizeof(float));
	cudaMemcpy(deviceScore, cuScore, total * sizeof(float), cudaMemcpyHostToDevice);
	cudaFreeHost(cuScore);
	return deviceScore;
}

BG* moveBgToCUDA(BinaryGrammar bg) {
	BG* grammars;
	cudaMallocHost((void**)&grammars, bg.size() * sizeof(BG));
	for (int i = 0; i < bg.size(); i++) {
		tuple<int, int, int> pair = get<0>(bg[i]);
		int symbol = get<0>(pair);
		int lsym = get<1>(pair);
		int rsym = get<2>(pair);
		float rulescore = get<1>(bg[i]);
		grammars[i].A = symbol;
		grammars[i].B = lsym;
		grammars[i].C = rsym;
		grammars[i].score = rulescore;
	}
	// allocate CUDA memory
	BG* gr;
	cudaMalloc((void**)&gr, bg.size() * sizeof(BG));
	cudaMemcpy(gr, grammars, bg.size() * sizeof(BG), cudaMemcpyHostToDevice);
	cudaFreeHost(grammars);
	return gr;
}

UG* moveUgToCUDA(UnaryGrammar ug) {
	UG* grammars;
	cudaMallocHost((void**)&grammars, ug.size() * sizeof(UG));
	for (int i = 0; i < ug.size(); i++) {
		tuple<int, int> pair = get<0>(ug[i]);
		int symbol = get<0>(pair);
		int target = get<1>(pair);
		float rulescore = get<1>(ug[i]);
		grammars[i].A = symbol;
		grammars[i].B = target;
		grammars[i].score = rulescore;
	}
	// allocate CUDA memory
	UG* gr;
	cudaMalloc((void**)&gr, ug.size() * sizeof(UG));
	cudaMemcpy(gr, grammars, ug.size() * sizeof(UG), cudaMemcpyHostToDevice);
	cudaFreeHost(grammars);
	return gr;
}

int* moveIntArrToCUDA(int* arr, int len) {
	int* arr_cu;
	cudaMalloc((void**)&arr_cu, len * sizeof(int));
	cudaMemcpy(arr_cu, arr, len * sizeof(int), cudaMemcpyHostToDevice);
	return arr_cu;
}

float* moveFloatArrToCUDA(float* arr, int len) {
	float* arr_cu;
	cudaMalloc((void**)&arr_cu, len * sizeof(float));
	cudaMemcpy(arr_cu, arr, len * sizeof(float), cudaMemcpyHostToDevice);
	return arr_cu;
}
