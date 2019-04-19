#include <iostream>
#include <tuple>
#include <vector>
#include <ctime>
#include "grammar.h"
using namespace std;

Scores initScores(int nWords, unordered_map<string, double> symbols) {
  Scores scores(nWords + 1, vector<unordered_map<string, double>>(nWords + 1, symbols));
  return scores;
}

void lexiconScores(Scores& scores, vector<string> sen, int nWords,
  unordered_map<string, vector<tuple<string, vector<double>>>> lex) {
  for(int start = 0; start < nWords; start++) {
    string word = sen[start];
    vector<tuple<string, vector<double>>> rules = lex[word];
    for(int i = 0; i < rules.size(); i++) {
      // Extract information from grammar rules
      tuple<string, vector<double>> pair = rules[i];
      string tag = get<0>(pair);
      vector<double> probs = get<1>(pair);
      for(int j = 0; j < probs.size(); j++) {
        string subtag = tag + '_' + to_string(j);
        scores[start][start+1][subtag] = probs[j];
      }
    }
  }
}

void binaryRelax(Scores& scores, int nWords, int length, BinaryGrammar gr) {
  for(int start = 0; start <= nWords-length; start++) {
    int end = start + length;
    for(int i = 0; i < gr.size(); i++) {
      // Extract information from grammar rules
      tuple<string, string, string> pair = get<0>(gr[i]);
      string symbol = get<0>(pair);
      string lsym = get<1>(pair);
      string rsym = get<2>(pair);
      double rulescore = get<1>(gr[i]);
      double current = scores[start][end][symbol];
      for(int split = start+1; split <= end-1; split++) {
        double lscore = scores[start][split][lsym];
        if (lscore > -DBL_MAX) {
          double rscore = scores[split][end][rsym];
          if (rscore > -DBL_MAX) {
            double total = rulescore + lscore + rscore;
            if (total > current) {
              current = total;
            }
          }
        }
      }
      scores[start][end][symbol] = current;
    }
  }
}

void unaryRelax(Scores& scores, int nWords, int length, UnaryGrammar gr) {
  for(int start = 0; start <= nWords-length; start++) {
    int end = start + length;
    for(int i = 0; i < gr.size(); i++) {
      // Extract information from grammar rules
      tuple<string, string> pair = get<0>(gr[i]);
      string symbol = get<0>(pair);
      string lsym = get<1>(pair);
      double rulescore = get<1>(gr[i]);
      double current = scores[start][end][symbol];
      if(scores[start][end][lsym] > -DBL_MAX) {
        double total = rulescore + scores[start][end][lsym];
        if (total > current) {
          scores[start][end][symbol] = total;
        }
      }
    }
  }
}

Ptree* searchHighest (Scores& scores, string symbol, vector<string> sen, int start, int end, BinaryGrammar gr2, UnaryGrammar gr1){
  Ptree* root = (Ptree*) malloc(sizeof(Ptree));
  if (symbol == "ATROOT") {
    double max = -DBL_MAX;
    string current = "start";
    for (auto it : scores[0][end]) {
      if (it.second > max) {
        max = it.second;
        current = it.first;
      }
    }
    root->symbol = current;
  } else {
    root->symbol = symbol;
  }
  Ptree* curr = root;
  for(int i = 0; i < gr1.size(); i++) {
    tuple<string, string> pair = get<0>(gr1[i]);
    string symbol = get<0>(pair);
    if (symbol == curr->symbol) {
      string lsym = get<1>(pair);
      double prob = get<1>(gr1[i]);
      if(scores[start][end][symbol] == scores[start][end][lsym] + prob){
        Ptree* child = (Ptree*) malloc(sizeof(Ptree));
        child->symbol = lsym;
        root->left = child;
        curr = child;
      }
    }
  }
  if (start+1 == end) {
    Ptree* leaf = (Ptree*) malloc(sizeof(Ptree));
    leaf->symbol = sen[start];
    curr->left = leaf;
    return root;
  }
  for(int j = 0; j < gr2.size(); j++) {
    tuple<string, string, string> pair = get<0>(gr2[j]);
    string symbol = get<0>(pair);
    if (symbol == curr->symbol) {
      string lsym = get<1>(pair);
      string rsym = get<2>(pair);
      double rscore= get<1>(gr2[j]);
      for(int split = start+1; split <= end-1; split++) {
        if(scores[start][end][symbol] == scores[start][split][lsym]+scores[split][end][rsym]+rscore){
          curr->left = searchHighest(scores, lsym, sen, start, split, gr2, gr1);
          curr->right = searchHighest(scores, rsym, sen, split, end, gr2, gr1);
        }
      }
    }
  }
  return root;
}

Ptree* parse(vector<string> sen, unordered_map<string, vector<tuple<string, vector<double>>>> lex,
   BinaryGrammar bg, UnaryGrammar ug, unordered_map<string, double> symbols) {
  int nWords = sen.size();
  Scores scores = initScores(nWords, symbols);
  lexiconScores(scores, sen, nWords, lex);
  for(int spanlen = 2; spanlen <= nWords; spanlen++) {
    binaryRelax(scores, nWords, spanlen, bg);
    unaryRelax(scores, nWords, spanlen, ug);
  }
  Ptree* result = searchHighest(scores, "ATROOT", sen, 0, nWords, bg, ug);
  return result;
}

int main(){
  // read grammar, lexicons, stentences and symbols
  BinaryGrammar bg = read_binary_grammar();
  UnaryGrammar ug = read_unary_grammar();
  unordered_map<string, vector<tuple<string, vector<double>>>> lexicons = read_lexicon();
  vector<vector<string>> sentences = read_sentences();
  unordered_map<string, double> symbols = read_symbols();

  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < 1; i++){
    parse(sentences[i], lexicons, bg, ug, symbols);
  }
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";
  /*
  unordered_map<string, double> symbols = ?
  for sen in stentences
    ptree = parse(sen, lexicons, gr2, gr1, symbols);

  */
}
