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

LR* moveLrToCUDA(unordered_map<string, vector<tuple<string, vector<float>>>> lex, WordToIdx wti, SymToIdx sti, int lr_size) {
	LR* lexrules;
	cudaMallocHost((void**)&lexrules, lr_size * sizeof(LR));
	int curr = 0;
	for (auto it:lex){
		string w = it.first;
		int wIdx = wti[w];
		vector<tuple<string, vector<float>>> rules = lex[w];
		for(int i = 0; i < rules.size(); i++) {
			tuple<string, vector<float>> pair = rules[i];
			string tag = get<0>(pair);
			vector<float> probs = get<1>(pair);
			for(int j = 0; j < probs.size(); j++) {
				string subtag = tag + '_' + to_string(j);
				int tagidx = sti[subtag];
				lexrules[curr].A = wIdx;
				lexrules[curr].B = tagidx;
				lexrules[curr].score = probs[j];
				curr++;
			}
		}
	}
	// allocate CUDA memory
	LR* lr;
	cudaMalloc((void**)&lr, lr_size * sizeof(LR));
	cudaMemcpy(lr, lexrules, lr_size * sizeof(LR), cudaMemcpyHostToDevice);
	cudaFreeHost(lexrules);
	return lr;
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
	cudaFree(arr);
	return arr_cu;
}

float* moveFloatArrToCUDA(float* arr, int len) {
	float* arr_cu;
	cudaMalloc((void**)&arr_cu, len * sizeof(float));
	cudaMemcpy(arr_cu, arr, len * sizeof(float), cudaMemcpyHostToDevice);
	cudaFree(arr);
	return arr_cu;
}

int countLex(unordered_map<string, vector<tuple<string, vector<float>>>> lex) {
	int total = 0;
	for (auto it:lex){
		string w = it.first;
		vector<tuple<string, vector<float>>> rules = lex[w];
		for(int i = 0; i < rules.size(); i++) {
			tuple<string, vector<float>> pair = rules[i];
			vector<float> probs = get<1>(pair);
			total += probs.size();
		}
	}
	return total;
}

__global__ void WordLexicon(float* score_arr, int start, LR* lr, int lr_size, int dim, int wIdx) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < lr_size) {
		int word = lr[threadId].A;
		if (word == wIdx){
			int subtag = lr[threadId].B;
			float prob = lr[threadId].score;
			score_arr[subtag*dim*dim + (start+1)*dim + start] = prob;
		}
	}
}

void lexiconCUDA(float* score_arr, vector<string> sen, int nWords, LR* lr, int lr_size, SymToIdx sti, IdxToSym its, WordToIdx wti) {
	int dim = nWords + 1;
	int start;
	for (start = 0; start < nWords; start++){
		int wIdx = wti[sen[start]];
		WordLexicon<<<((lr_size+NUM_THREADS)/NUM_THREADS), NUM_THREADS>>>(score_arr, start, lr, lr_size, dim, wIdx);
	}
}
