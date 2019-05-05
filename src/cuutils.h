#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#include "grammar.h"

typedef struct {
  int A;
  int B;
  int C;
  float score;
} BG;

typedef struct {
  int A;
  int B;
  float score;
} UG;

float* moveScoreToCUDA(Scores s, int dim1, int dim2, int dim3);

BG* moveBgToCUDA(BinaryGrammar bg);

UG* moveUgToCUDA(UnaryGrammar ug);

int* moveIntArrToCUDA(int* arr, int len);

float* moveFloatArrToCUDA(float* arr, int len);

#endif
