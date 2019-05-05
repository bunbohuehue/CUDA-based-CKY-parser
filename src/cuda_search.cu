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

__global__ void findRoot(float* score_arr, int* root_CUDA, int dim1, int dim2, int dim3) {
	float max = -FLT_MAX;
	int current = 0;
	for (int i = 0; i < dim3; i++) {
		if (score_arr[i*dim1*dim2+(dim2-1)*dim1] > max) {
			max = score_arr[i*dim1*dim2+(dim2-1)*dim1];
			current = i;
		}
	}
	root_CUDA[0] = current;
}

__global__ void searchUnaryRules (float* score_arr, int* lsym_CUDA, UG* ug, int ug_size,
	int dim1, int dim2, int dim3, int start, int end, int sym) {
		int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId < ug_size){
			int symbol = ug[threadId].A;
			if (symbol == sym) {
				int lsym = ug[threadId].B;
				float rulescore = ug[threadId].score;
				float symbolscore = score_arr[symbol*dim1*dim2 + end*dim1 + start];
				float lsymscore = score_arr[lsym*dim1*dim2 + end*dim1 + start];
				float diff = symbolscore - lsymscore - rulescore;
				if(diff > -TOLERANCE && diff < TOLERANCE){
					lsym_CUDA[0] = lsym;
				}
			}
		}
}

__global__ void searchBinaryRules (float* score_arr, int* children_CUDA, BG* bg, int bg_size,
	int dim1, int dim2, int dim3, int start, int end, int sym, int* flag) {
		int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId < bg_size) {
			int symbol = bg[threadId].A;
			if (symbol == sym) {
				int lsym = bg[threadId].B;
				int rsym = bg[threadId].C;
				float rulescore = bg[threadId].score;
				for(int split = start+1; split <= end-1; split++) {
					float symbolscore = score_arr[symbol*dim1*dim2 + end*dim1 + start];
					float lsymscore = score_arr[lsym*dim1*dim2 + split*dim1 + start];
					float rsymscore = score_arr[rsym*dim1*dim2 + end*dim1 + split];
					float diff = symbolscore - lsymscore -  rsymscore - rulescore;
					if(diff > -TOLERANCE && diff < TOLERANCE){
						if (atomicCAS(flag, 0, 1) == 0){
							children_CUDA[0] = lsym;
							children_CUDA[1] = rsym;
							children_CUDA[2] = split;
						}
					}
				}
			}
		}
}

Ptree* CUDAsearch (float* score_arr, int symidx, vector<string> sen, int start, int end,
	BG* bg, int bg_size, UG* ug, int ug_size, int dim1, int dim2, int dim3, SymToIdx sti, IdxToSym its){
	Ptree* root = (Ptree*) malloc(sizeof(Ptree));
	// Find root if at depth 0
	if (symidx == -1) {
		int* rooti = new int[1];
		rooti[0] = -1;
		int* root_CUDA;
		cudaMalloc((void**)&root_CUDA, sizeof(int));
		cudaMemcpy(root_CUDA, rooti, sizeof(int), cudaMemcpyHostToDevice);
		findRoot<<<1,1>>>(score_arr, root_CUDA, dim1, dim2, dim3);
		cudaMemcpy(rooti, root_CUDA, sizeof(int), cudaMemcpyDeviceToHost);
		root->symbol = its[rooti[0]];
		cudaFree(root);
		cudaFree(root_CUDA);
	} else {
		root->symbol = its[symidx];
	}
	Ptree* curr = root;
	//cout << root->symbol << endl;
	int* lsym = new int[1];
	lsym[0] = -1;
	int* lsym_CUDA;
	cudaMalloc((void**)&lsym_CUDA, sizeof(int));
	cudaMemcpy(lsym_CUDA, lsym, sizeof(int), cudaMemcpyHostToDevice);
	searchUnaryRules<<<((ug_size+NUM_THREADS)/NUM_THREADS), NUM_THREADS>>>(score_arr, lsym_CUDA, ug, ug_size,
		dim1, dim2, dim3, start, end, sti[curr->symbol]);
	cudaMemcpy(lsym, lsym_CUDA, sizeof(int), cudaMemcpyDeviceToHost);
	if (lsym[0] != -1) {
		Ptree* child = (Ptree*) malloc(sizeof(Ptree));
		child->symbol = its[lsym[0]];
		root->left = child;
		curr = child;
	}
	cudaFree(lsym);
	cudaFree(lsym_CUDA);

	if (start+1 == end) {
		struct Ptree *leafnode;
   	leafnode = new struct Ptree;
		leafnode->symbol = sen[start];
		curr->left = leafnode;
		return root;
	}

	int* children = new int[3];
	children[0] = -1;
	children[1] = -1;
	children[2] = -1;
	int* children_CUDA;
	cudaMalloc((void**)&children_CUDA, 3 * sizeof(int));
	cudaMemcpy(children_CUDA, children, 3 * sizeof(int), cudaMemcpyHostToDevice);
	int* modified = new int[1];
	modified[0] = 0;
	int* modified_CUDA;
	cudaMalloc((void**)&modified_CUDA, sizeof(int));
	cudaMemcpy(modified_CUDA, modified, sizeof(int), cudaMemcpyHostToDevice);
	searchBinaryRules<<<((bg_size+NUM_THREADS)/NUM_THREADS), NUM_THREADS>>>(score_arr, children_CUDA, bg, bg_size,
		dim1, dim2, dim3, start, end, sti[curr->symbol], modified_CUDA);
	cudaMemcpy(children, children_CUDA, 3 * sizeof(int), cudaMemcpyDeviceToHost);
	if (children[0] != -1 && children[1] != -1 && children[2] != -1) {
		curr->left = CUDAsearch(score_arr, children[0], sen, start, children[2],
			bg, bg_size, ug, ug_size, dim1, dim2, dim3, sti, its);
		curr->right = CUDAsearch(score_arr, children[1], sen, children[2], end,
			bg, bg_size, ug, ug_size, dim1, dim2, dim3, sti, its);
	}
	cudaFree(children);
	cudaFree(children_CUDA);
	cudaFree(modified);
	cudaFree(modified_CUDA);
	return root;
}
