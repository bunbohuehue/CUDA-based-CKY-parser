#ifndef GRAMMAR_H
#define GRAMMAR_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <float.h>
#include <unordered_map>
using namespace std;

/* inside the first tuple: Rule A -> B C, and the score of it */
typedef vector<tuple<tuple<int, int, int>, float>> BinaryGrammar;
/* inside the first tuple: Rule A -> B, and the score of it */
typedef vector<tuple<tuple<int, int>, float>> UnaryGrammar;
/* 1d: Word. 2d: Tag. 3d: array of scores of subtags */
typedef unordered_map<string, vector<tuple<string, vector<float>>>> Lexicons;
/* score array */
typedef vector<vector<vector<float>>> Scores;
typedef vector<vector<bool>> Occured;

/* for index <=> symbol conversion */
typedef unordered_map<string, int> SymToIdx;
typedef unordered_map<int, string> IdxToSym;
typedef unordered_map<string, int> WordToIdx;
typedef vector<string> Symbols;

typedef unordered_map<int, vector<tuple<int, int, float>>> BinaryGrammar_SYM;
typedef unordered_map<int, vector<tuple<int, float>>> UnaryGrammar_SYM;

/* Read binary grammars */
BinaryGrammar read_binary_grammar(SymToIdx sti, BinaryGrammar_SYM& bg);

/* Read unary grammars */
UnaryGrammar read_unary_grammar(SymToIdx sti, UnaryGrammar_SYM& ug);

/* Read lexicon scores */
unordered_map<string, vector<tuple<string, vector<float>>>> read_lexicon(SymToIdx sti, WordToIdx& wti);

/* Read input sentences to be parsed */
vector<vector<string>> read_sentences();

int generate_sym_to_rules_b(BinaryGrammar_SYM bg, int*& rule_arr, float*& score_arr, int*& lens, int*& syms);

int generate_sym_to_rules_u(UnaryGrammar_SYM ug, int*& rule_arr, float*& score_arr, int*& lens, int*& syms);

/* Read all grammar symbols into dict form */
int read_symbols(SymToIdx& sti, IdxToSym& its);

struct Ptree {
  string symbol;
  Ptree* left;
  Ptree* right;
};

#endif
