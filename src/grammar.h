#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <float.h>
using namespace std;

/* inside the first tuple: Rule A -> B C, and the score of it */
typedef vector<tuple<tuple<string, string, string>, double>> BinaryGrammar;
/* inside the first tuple: Rule A -> B, and the score of it */
typedef vector<tuple<tuple<string, string>, double>> UnaryGrammar;
/* 1d: Word. 2d: Tag. 3d: array of scores of subtags */
typedef vector<tuple<string, string, vector<double>>> Lexicons;
/* score array */
typedef vector<vector<unordered_map<string, double>>> Scores;

/* Read binary grammars */
BinaryGrammar read_binary_grammar();

/* Read unary grammars */
UnaryGrammar read_unary_grammar();

/* Read lexicon scores */
Lexicons read_lexicon();
