#ifndef __CUDA_PARSER_H__
#define __CUDA_PARSER_H__

#include "grammar.h"
#include "cuutils.h"

void parseAllRuleBased (vector<vector<string>> sen, unordered_map<string, vector<tuple<string, vector<float>>>> lex,
		 BinaryGrammar bg, UnaryGrammar ug, int num_symbol, SymToIdx sti, IdxToSym its, int num_sen, WordToIdx wti);

void parseAllBlockBased (vector<vector<string>> sen, unordered_map<string, vector<tuple<string, vector<float>>>> lex,
		 BinaryGrammar bg, UnaryGrammar ug, int num_symbol, SymToIdx sti, IdxToSym its, int num_sen, int num_blocks_b, int num_blocks_u,
     int* gr_b, int* gr_u, int* lens_b, int* lens_u, int* syms_b, int* syms_u, float* score_b, float* score_u, WordToIdx wti);

void lexiconCUDA(float* score_arr, vector<string> sen, int nWords, LR* lr, int lr_size, SymToIdx sti, IdxToSym its, WordToIdx wti);

Ptree* CUDAsearch (float* score_arr, int symidx, vector<string> sen, int start, int end,
	   BG* bg, int bg_size, UG* ug, int ug_size, int dim1, int dim2, int dim3, SymToIdx sti, IdxToSym its);

Ptree* parse_sequential(vector<string> sen, unordered_map<string, vector<tuple<string, vector<float>>>> lex,
      BinaryGrammar bg, UnaryGrammar ug, int num_symbol, SymToIdx sti, IdxToSym its);

Ptree* RuleBasedParse(vector<string> sen, unordered_map<string, vector<tuple<string, vector<float>>>> lex,
      BinaryGrammar bg, UnaryGrammar ug, int num_symbol, SymToIdx sti, IdxToSym its);

void lexiconScores(Scores& scores, vector<string> sen, int nWords, unordered_map<string,
      vector<tuple<string, vector<float>>>> lex, SymToIdx sti, IdxToSym its, Occured& occured);

Scores initScores(int nWords, int num_symbol);

#endif
