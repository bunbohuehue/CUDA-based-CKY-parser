#ifndef __CUDA_PARSER_H__
#define __CUDA_PARSER_H__

#include "grammar.h"

typedef struct {
  int A;
  int B;
  int C;
  float score;
} BG;

Ptree* parse(vector<string> sen, unordered_map<string, vector<tuple<string, vector<float>>>> lex,
       BinaryGrammar bg, UnaryGrammar ug, int num_symbol, SymToIdx sti, IdxToSym its);

#endif
