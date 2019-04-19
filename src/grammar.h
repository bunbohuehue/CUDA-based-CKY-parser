#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <float.h>
#include <unordered_map>
using namespace std;

/* inside the first tuple: Rule A -> B C, and the score of it */
typedef vector<tuple<tuple<string, string, string>, double>> BinaryGrammar;
/* inside the first tuple: Rule A -> B, and the score of it */
typedef vector<tuple<tuple<string, string>, double>> UnaryGrammar;
/* 1d: Word. 2d: Tag. 3d: array of scores of subtags */
typedef unordered_map<string, vector<tuple<string, vector<double>>>> Lexicons;
/* score array */
typedef vector<vector<unordered_map<string, double>>> Scores;

/* Read binary grammars */
BinaryGrammar read_binary_grammar();

/* Read unary grammars */
UnaryGrammar read_unary_grammar();

/* Read lexicon scores */
Lexicons read_lexicon();

/* Read input sentences to be parsed */
vector<vector<string>> read_sentences();

/* Read all grammar symbols into dict form */
unordered_map<string, double> read_symbols();

struct Ptree {
  string symbol;
  Ptree* left;
  Ptree* right;
};
