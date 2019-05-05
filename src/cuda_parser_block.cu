#define NUM_THREADS 1024
#define TOLERANCE 0.001
#include <iostream>
#include <tuple>
#include <vector>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "cuda_parser.h"
using namespace std;

__device__ __forceinline__ float atomicMaxFloat_b (float* addr, float value) {
	float old;
	old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
		__uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
	return old;
}

__global__ void update_score_b(float* score, float*shared_max, int dim1, int dim2, int num_symbol, int num_starts, int spanlen) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < num_symbol) {
		for (int i = 0; i < num_starts; i++) {
			int end = i + spanlen;
				if (score[threadId*dim1*dim2 + end*dim1 + i] < shared_max[i*num_symbol+threadId]){
					score[threadId*dim1*dim2 + end*dim1 + i] = shared_max[i*num_symbol+threadId];
				}
		}
	}
}

__global__ static void BlockUnaryRelaxKernel(float* deviceScores, float* shared_max,
	int dim1, int dim2, int dim3, int num_starts, int spanlen, int num_blocks, int* gr_u,
	int* lens_u, int* syms_u, float* score_u) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	int idx_to_row = threadId / NUM_THREADS;
	int idx_to_col = threadId % NUM_THREADS;

	int symbol = syms_u[idx_to_row];
	int num_rules = lens_u[idx_to_row];
	int lsym = gr_u[idx_to_row * NUM_THREADS + idx_to_col];
	float score = score_u[idx_to_row * NUM_THREADS + idx_to_col];

  if (idx_to_col < num_rules) {
    for (int start = 0; start < num_starts; start++) {
  		float localMax = -FLT_MAX;
  		int end = start + spanlen;
  		// score_arr is on CUDA
      float lscore;
      lscore = deviceScores[lsym*dim1*dim2 + end*dim1 + start];
      if (lscore > -FLT_MAX) {
        float total = score + lscore;
        if (total > localMax) {
          localMax = total;
        }
      }
      atomicMaxFloat_b(&shared_max[start*dim3+symbol], localMax);
  	}
  }
}

__global__ static void BlockBinaryRelaxKernel(float* deviceScores, float* shared_max,
	int dim1, int dim2, int dim3, int num_starts, int spanlen, int num_blocks, int* gr_b,
	int* lens_b, int* syms_b, float* score_b) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	int idx_to_row = threadId / NUM_THREADS;
	int idx_to_col = threadId % NUM_THREADS;

	int symbol = syms_b[idx_to_row];
	int num_rules = lens_b[idx_to_row];
	int lsym = gr_b[idx_to_row * 2 * NUM_THREADS + 2 * idx_to_col];
	int rsym = gr_b[idx_to_row * 2 * NUM_THREADS + 2 * idx_to_col + 1];
	float score = score_b[idx_to_row * NUM_THREADS + idx_to_col];

  // only proceed if the rule corresponding to threadid is actually a rule
  if (idx_to_col < num_rules) {
    for (int start = 0; start < num_starts; start++) {
  		float localMax = -FLT_MAX;
  		int end = start + spanlen;
  		// score_arr is on CUDA
  		for (int split = start + 1; split <= end - 1; split++) {
  			// [start][split][lsym]
  			float lscore;
  			lscore = deviceScores[lsym*dim1*dim2 + split*dim1 + start];
  			// [split][end][rsym]
  			float rscore;
  			rscore = deviceScores[rsym*dim1*dim2 + end*dim1 + split];
  			if (lscore > -FLT_MAX && rscore > -FLT_MAX) {
  				float total;
  				total = score + lscore + rscore;
  				if (total > localMax) {
  					localMax = total;
  				}
  			}
  		}
  		atomicMaxFloat_b(&shared_max[start*dim3+symbol], localMax);
  	}
  }
}

void BlockBasedUnaryRelax (float* score_arr, int nWords, int length, int num_symbol,
	int num_blocks, int* gr_u, int* lens_u, int* syms_u, float* score_u) {
	int dim1 = nWords + 1;
	int dim2 = nWords + 1;

  int num_starts = nWords - length + 1;
	int num_ele = num_symbol * num_starts;
	float* shared_max = new float[num_ele];
	for (int i = 0; i < num_ele; i++) {
		shared_max[i] = -FLT_MAX;
	}
	float* shared_max_CUDA;
	cudaMalloc((void**)&shared_max_CUDA, num_ele * sizeof(float));
	cudaMemcpy(shared_max_CUDA, shared_max, num_ele * sizeof(float), cudaMemcpyHostToDevice);
	cudaFree(shared_max);

	BlockUnaryRelaxKernel<<<num_blocks, NUM_THREADS>>>(score_arr, shared_max_CUDA, nWords+1,
		nWords+1, num_symbol, num_starts, length, num_blocks, gr_u, lens_u, syms_u, score_u);

	update_score_b<<<((num_symbol+NUM_THREADS)/NUM_THREADS), NUM_THREADS>>>(score_arr, shared_max_CUDA, dim1, dim2, num_symbol, num_starts, length);
	cudaFree(shared_max_CUDA);
}

void BlockBasedBinaryRelax (float* score_arr, int nWords, int length, int num_symbol,
	int num_blocks, int* gr_b, int* lens_b, int* syms_b, float* score_b) {
	int dim1 = nWords + 1;
	int dim2 = nWords + 1;

	int num_starts = nWords - length + 1;
	int num_ele = num_symbol * num_starts;
	float* shared_max = new float[num_ele];
	for (int i = 0; i < num_ele; i++) {
		shared_max[i] = -FLT_MAX;
	}
	float* shared_max_CUDA;
	cudaMalloc((void**)&shared_max_CUDA, num_ele * sizeof(float));
	cudaMemcpy(shared_max_CUDA, shared_max, num_ele * sizeof(float), cudaMemcpyHostToDevice);
	cudaFree(shared_max);

	BlockBinaryRelaxKernel<<<num_blocks, NUM_THREADS>>>(score_arr, shared_max_CUDA, nWords+1,
		nWords+1, num_symbol, num_starts, length, num_blocks, gr_b, lens_b, syms_b, score_b);

	update_score_b<<<((num_symbol+NUM_THREADS)/NUM_THREADS), NUM_THREADS>>>(score_arr, shared_max_CUDA, dim1, dim2, num_symbol, num_starts, length);
	cudaFree(shared_max_CUDA);
}

Ptree* BlockBasedParse(vector<string> sen, unordered_map<string, vector<tuple<string, vector<float>>>> lex,
			 int num_symbol, SymToIdx sti, IdxToSym its, BG* gr1, UG* gr2, int bg_size, int ug_size, int num_blocks_b, int num_blocks_u,
		 	 int* gr_b, int* gr_u, int* lens_b, int* lens_u, int* syms_b, int* syms_u, float* score_b, float* score_u) {

	int nWords = (int)sen.size();
	Scores scores = initScores(nWords, num_symbol);
	Occured occured(nWords, vector<bool>(num_symbol, 0));
	lexiconScores(scores, sen, nWords, lex, sti, its, occured);

	float* score_arr = moveScoreToCUDA(scores, nWords + 1, nWords + 1, num_symbol);

	for(int spanlen = 2; spanlen <= nWords; spanlen++) {
		BlockBasedBinaryRelax(score_arr, nWords, spanlen, num_symbol,
			num_blocks_b, gr_b, lens_b, syms_b, score_b);
		BlockBasedUnaryRelax(score_arr, nWords, spanlen, num_symbol,
			num_blocks_u, gr_u, lens_u, syms_u, score_u);
	}

	int dim1 = nWords + 1;
	int dim2 = nWords + 1;
	int dim3 = num_symbol;

  Ptree* result = CUDAsearch(score_arr, -1, sen, 0, nWords, gr1, bg_size, gr2, ug_size, dim1,dim2,dim3, sti, its);
  cout << result->symbol << endl;
  cudaFree(score_arr);
  return result;
}

void parseAllBlockBased (vector<vector<string>> sen, unordered_map<string, vector<tuple<string, vector<float>>>> lex,
		 BinaryGrammar bg, UnaryGrammar ug, int num_symbol, SymToIdx sti, IdxToSym its, int num_sen, int num_blocks_b, int num_blocks_u,
     int* gr_b, int* gr_u, int* lens_b, int* lens_u, int* syms_b, int* syms_u, float* score_b, float* score_u) {
	int total = 0;
	int num = 0;
	int bg_size = bg.size();
	int ug_size = ug.size();
  BG* gr1 = moveBgToCUDA(bg);
	UG* gr2 = moveUgToCUDA(ug);
	int* gr_b_cu = moveIntArrToCUDA(gr_b, num_blocks_b * 2048);
	int* gr_u_cu = moveIntArrToCUDA(gr_u, num_blocks_u * 1024);
	int* lens_b_cu = moveIntArrToCUDA(lens_b, num_blocks_b);
	int* lens_u_cu = moveIntArrToCUDA(lens_u, num_blocks_u);
	int* syms_b_cu = moveIntArrToCUDA(syms_b, num_blocks_b);
	int* syms_u_cu = moveIntArrToCUDA(syms_u, num_blocks_u);
	float* score_b_cu = moveFloatArrToCUDA(score_b, num_blocks_b);
	float* score_u_cu = moveFloatArrToCUDA(score_u, num_blocks_u);
	for (int i = 0; i < num_sen; i++){
		int len = (int)sen[i].size();
		num += 1;
		total += len;
		// BlockBasedParse(sen[i], lex, gr1, gr2, num_symbol, sti, its, bg_size, ug_size, bg, ug);
		BlockBasedParse(sen[i], lex, num_symbol, sti, its, gr1, gr2, bg_size, ug_size, num_blocks_b, num_blocks_u,
		gr_b_cu, gr_u_cu, lens_b_cu, lens_u_cu, syms_b_cu, syms_u_cu, score_b_cu, score_u_cu);
		cout << "Finished parsing sentence (CUDA) " << num << endl;
	}
	std::cout << "avg len: " << total/num_sen << " \n";
}
