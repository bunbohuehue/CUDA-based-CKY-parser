#ifndef __CUDA_PARSER_H__
#define __CUDA_PARSER_H__

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

Ptree* parse_sequential(vector<string> sen, unordered_map<string, vector<tuple<string, vector<float>>>> lex,
       BinaryGrammar bg, UnaryGrammar ug, int num_symbol, SymToIdx sti, IdxToSym its);

Ptree* parse(vector<string> sen, unordered_map<string, vector<tuple<string, vector<float>>>> lex,
       BinaryGrammar bg, UnaryGrammar ug, int num_symbol, SymToIdx sti, IdxToSym its);

#endif
