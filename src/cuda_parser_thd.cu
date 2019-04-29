#define NUM_THREADS 1024
#include <iostream>
#include <tuple>
#include <vector>
#include <ctime>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cuda_parser.h"
using namespace std;

/* Below are all functions used in sequential version of the parser
   These will also be useful when implementing threadBasedBinaryRelax
 */

Scores initScores (int nWords, int num_symbol) {
	Scores scores(nWords + 1, vector<vector<float>>(nWords + 1, vector<float>(num_symbol, -FLT_MAX)));
	return scores;
}

void lexiconScores (Scores& scores, vector<string> sen, int nWords, unordered_map<string,
					vector<tuple<string, vector<float>>>> lex, SymToIdx sti, IdxToSym its, Occured& occured) {
	for(int start = 0; start < nWords; start++) {
		string word = sen[start];
		vector<tuple<string, vector<float>>> rules = lex[word];
		for(int i = 0; i < rules.size(); i++) {
			// Extract information from grammar rules
			tuple<string, vector<float>> pair = rules[i];
			string tag = get<0>(pair);
			vector<float> probs = get<1>(pair);
			for(int j = 0; j < probs.size(); j++) {
				string subtag = tag + '_' + to_string(j);
				int tagidx = sti[subtag];
				scores[start][start+1][tagidx] = probs[j];
				occured[0][tagidx] = 1;
			}
		}
	}
}

void binaryRelax (Scores& scores, int nWords,
				  int length, BinaryGrammar gr, Occured& occured) {
	for(int i = 0; i < gr.size(); i++) {
		// Extract information from grammar rules
		tuple<int, int, int> pair = get<0>(gr[i]);
		int symbol = get<0>(pair);
		int lsym = get<1>(pair);
		int rsym = get<2>(pair);
		float rulescore = get<1>(gr[i]);
		for(int split = 1; split < length; split++) {
			if (occured[split-1][lsym] && occured[length-split-1][rsym]){
				for(int start = 0; start <= nWords-length; start++) {
					int end = start + length;
					float lscore = scores[start][start+split][lsym];
					if (lscore > -FLT_MAX) {
						float rscore = scores[start+split][end][rsym];
						if (rscore > -FLT_MAX) {
							float current = scores[start][end][symbol];
							float total = rulescore + lscore + rscore;
							if (total > current) {
								scores[start][end][symbol] = total;
								occured[length-1][symbol] = 1;
							}
						}
					}
				}
			}
		}
	}
}

void unaryRelax (Scores& scores, int nWords,
				 int length, UnaryGrammar gr, Occured& occured) {
	for(int i = 0; i < gr.size(); i++) {
		// Extract information from grammar rules
		tuple<int, int> pair = get<0>(gr[i]);
		int symbol = get<0>(pair);
		int lsym = get<1>(pair);
		if (occured[length-1][lsym]) {
			float rulescore = get<1>(gr[i]);
			for(int start = 0; start <= nWords-length; start++) {
				int end = start + length;
				float current = scores[start][end][symbol];
				if(scores[start][end][lsym] > -FLT_MAX) {
					float total = rulescore + scores[start][end][lsym];
					if (total > current) {
						scores[start][end][symbol] = total;
						occured[length-1][symbol] = 1;
					}
				}
			}
		}
	}
}

Ptree* searchHighest (Scores& scores, int symidx, vector<string> sen,
					  int start, int end, BinaryGrammar gr2, UnaryGrammar gr1, SymToIdx sti, IdxToSym its){
	Ptree* root = (Ptree*) malloc(sizeof(Ptree));
	if (symidx == -1) {
		float max = -FLT_MAX;
		int current = 0;
		for (int i = 0; i < scores[0][end].size(); i++) {
			if (scores[0][end][i] > max) {
				max = scores[0][end][i];
				current = i;
			}
		}
		root->symbol = its[current];
	} else {
		root->symbol = its[symidx];
	}
	Ptree* curr = root;
	for(int i = 0; i < gr1.size(); i++) {
		tuple<int, int> pair = get<0>(gr1[i]);
		int symbol = get<0>(pair);
		if (symbol == sti[curr->symbol]) {
			int lsym = get<1>(pair);
			float prob = get<1>(gr1[i]);
			if(scores[start][end][symbol] == scores[start][end][lsym] + prob){
				Ptree* child = (Ptree*) malloc(sizeof(Ptree));
				child->symbol = its[lsym];
				root->left = child;
				curr = child;
			}
		}
	}
	if (start+1 == end) {
		Ptree* leaf = (Ptree*) malloc(sizeof(Ptree));
		leaf->symbol = sen[start];
		curr->left = leaf;
		return root;
	}
	for(int j = 0; j < gr2.size(); j++) {
		tuple<int, int, int> pair = get<0>(gr2[j]);
		int symbol = get<0>(pair);
		if (symbol == sti[curr->symbol]) {
			int lsym = get<1>(pair);
			int rsym = get<2>(pair);
			float rscore= get<1>(gr2[j]);
			for(int split = start+1; split <= end-1; split++) {
				if(scores[start][end][symbol] == scores[start][split][lsym]+scores[split][end][rsym]+rscore){
					curr->left = searchHighest(scores, lsym, sen, start, split, gr2, gr1, sti, its);
					curr->right = searchHighest(scores, rsym, sen, split, end, gr2, gr1, sti, its);
				}
			}
		}
	}
	return root;
}

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
/*
__device__ static float atomicMax(float* address, float val) {
	int* address_as_i = (int*) address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}
*/
__device__ __forceinline__ float atomicMaxFloat (float* addr, float value) {
  float old;
  old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
		__uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
	return old;
}

__global__ static void check_shared_max(float* shared_max, int num_symbol) {
  for (int i = 0; i < num_symbol; i++) {
    if (shared_max[i] > -FLT_MAX) {
      printf("sldfjsdklfjdslkfjdsjldsfj\n");
      return;
    }
  }
  printf("ALL MINS\n");
}

__global__ static void check_device_score(float* device_scores, int dim1, int dim2, int dim3) {
	int ret = 0;
	for (int i = 0; i < dim1; i++){
		for (int j = 0; j < dim2; j++) {
			for (int k = 0; k < dim3; k++) {
				if(device_scores[k*dim1*dim2 + j*dim1 + i] != -FLT_MAX) {
					ret++;
				}
			}
		}
	}
  printf("UNZEROS: %d\n", ret);
}

__device__ static void check_score(float* device_scores, int dim1, int dim2, int dim3) {
	int ret = 0;
	for (int i = 0; i < dim1; i++){
		for (int j = 0; j < dim2; j++) {
			for (int k = 0; k < dim3; k++) {
				if(device_scores[k*dim1*dim2 + j*dim1 + i] != -FLT_MAX) {
					ret++;
				}
			}
		}
	}
  printf("UNZEROS: %d\n", ret);
}

__global__ static void BinaryRelaxKernel(BG* bg, float* deviceScores, float* shared_max,
  int rulesize, int dim1, int dim2, int dim3, int start, int end) {

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int symbol = bg[threadId].A;
	int lsym = bg[threadId].B;
	int rsym = bg[threadId].C;
	float rulescore = bg[threadId].score;
	// check_score(deviceScores, dim1, dim2, dim3);
	float localMax = -FLT_MAX;
  for (int split = start + 1; split <= end - 1; split++) {
		// [start][split][lsym]
    float lscore;
		lscore = deviceScores[lsym*dim1*dim2 + split*dim1 + start];
		// [split][end][rsym]
		float rscore;
		rscore = deviceScores[rsym*dim1*dim2 + end*dim1 + split];
		float total;
		total = rulescore + lscore + rscore;
		if (total > localMax) {
			printf("fuck your mother\n");
			localMax = total;
		}
  }
	atomicMaxFloat(&shared_max[symbol], localMax);
}

void RuleBasedBinaryRelax (Scores& scores, int nWords, int length, BG* bg, int bg_size, int num_symbol) {
  // Note that bg is already on device
  // begin of loop body
  for (int start = 0; start <= nWords - length; start++) {
		int end = start + length;

    // score_arr is on CUDA
    float* score_arr = moveScoreToCUDA(scores, nWords + 1, nWords + 1, num_symbol);
		// check_device_score<<<1, 1>>>(score_arr, nWords + 1, nWords + 1, num_symbol);
    float* shared_max = new float[num_symbol];
    for (int i = 0; i < num_symbol; i++) {
      shared_max[i] = -FLT_MAX;
    }
    float* shared_max_CUDA;
    cudaMalloc((void**)&shared_max_CUDA, num_symbol * sizeof(float));
    cudaMemcpy(shared_max_CUDA, shared_max, num_symbol * sizeof(float), cudaMemcpyHostToDevice);

		for (int i = 0; i < bg_size; i += NUM_THREADS) {
			if (i + NUM_THREADS < bg_size) {
				BinaryRelaxKernel<<<1, NUM_THREADS>>>(bg, score_arr, shared_max_CUDA, bg_size,
		      nWords+1, nWords+1, num_symbol, start, end);
			}
			else {
				BinaryRelaxKernel<<<1, (bg_size - i - 1)>>>(bg, score_arr, shared_max_CUDA, bg_size,
		      nWords+1, nWords+1, num_symbol, start, end);
			}
		}
    // check_shared_max<<<1, 1>>>(shared_max_CUDA, num_symbol);
    // copy back the shared_max array modified by kernel
    cudaMemcpy(shared_max, shared_max_CUDA, num_symbol * sizeof(float), cudaMemcpyDeviceToHost);
    // update score array
    for (int i = 0; i < num_symbol; i++) {
			if (scores[start][end][i] < shared_max[i]){
				scores[start][end][i] = shared_max[i];
			}
    }
		cudaFree(shared_max_CUDA);
		cudaFree(score_arr);
  }
}

Ptree* parse(vector<string> sen, unordered_map<string, vector<tuple<string, vector<float>>>> lex,
			 BinaryGrammar bg, UnaryGrammar ug, int num_symbol, SymToIdx sti, IdxToSym its) {
	int nWords = (int)sen.size();
	//std::cout << "total words: " << nWords << " \n";
	Scores scores = initScores(nWords, num_symbol);
	Occured occured(nWords, vector<bool>(num_symbol, 0));
	lexiconScores(scores, sen, nWords, lex, sti, its, occured);
  // copy grammar to device
  int bg_size = bg.size();
  BG* gr = moveBgToCUDA(bg);
	for(int spanlen = 2; spanlen <= nWords; spanlen++) {
		RuleBasedBinaryRelax(scores, nWords, spanlen, gr, bg_size, num_symbol);
    // TODO: ALSO IMPLEMENT CUDA VERSION OF UNARYRELAX
    // THIS WILL REDUCE THE AMOUNT OF MEMORY COPY DRAMATICALLY
    // AND ALL THE WORK COULD BE DONE ON CUDA DEVICE
		unaryRelax(scores, nWords, spanlen, ug, occured);
	}
	cudaFree(gr);
	Ptree* result = searchHighest(scores, -1, sen, 0, nWords, bg, ug, sti, its);
	std::cout << "Root symbol: " << result->symbol << "s\n";
	// std::cout << "Left child symbol: " << result->left->symbol << "s\n";
	// std::cout << "Right child symbol: " << result->right->symbol << "s\n";
	return result;
}
