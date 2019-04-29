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
typedef vector<string> Symbols;

/* Read binary grammars */
BinaryGrammar read_binary_grammar(SymToIdx sti);

/* Read unary grammars */
UnaryGrammar read_unary_grammar(SymToIdx sti);

/* Read lexicon scores */
Lexicons read_lexicon(SymToIdx sti);

/* Read input sentences to be parsed */
vector<vector<string>> read_sentences();

/* Read all grammar symbols into dict form */
int read_symbols(SymToIdx& sti, IdxToSym& its);

struct Ptree {
  string symbol;
  Ptree* left;
  Ptree* right;
};

#endif
