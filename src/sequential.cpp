#include <iostream>
#include <tuple>
#include <vector>
#include "grammar.h"
using namespace std;

Scores initScores(int nWords, unordered_map<string, double> symbols) {
  Scores scores(nWords, vector<unordered_map<string, double>>(nWords, symbols));
  return scores;
}

void lexiconScores(Scores scores, vector<string> sen, int nWords, Lexicons lex) {
  for(int start = 0; start < nWords; start++) {
    string w = sen[start];
    for(int i = 0; i < lex.size(); i++) {
      // Extract information from grammar rules
      tuple pair = gr[i]
      string word = pair.get(1)
      if (w == word) {
        string tag = pair.get(0)
        vector<double> probs = pair.get(2);
        for(int i = 0; i < probs.size(); i++) {
          string subtag = tag + '_' + to_string(i);
          scores[start][start+1][subtag] = probs[i];
        }
      }
    }
  }
}

void binaryRelax(Scores scores, int nWords, int length, BinaryGrammar gr) {
  for(int start = 0; start <= nWords-length; start++) {
    int end = start + length;
    for(int i = 0; i < gr.size(); i++) {
      // Extract information from grammar rules
      tuple pair = gr[i].get(0)
      string symbol = pair.get(0)
      string lsym = pair.get(1)
      string rsym = pair.get(2)
      double rulescore = gr[i].get(1)
      double current = scores[start][end][symbol];
      for(int split = start+1; split <= end-1; split++) {
        double lscore = scores[start][split][lsym];
        double rscore = scores[start][split][rsym];
        double total = rulescore + lscore + rscore;
        if (total > current) {
          current = total;
        }
      }
      scores[start][end][symbol] = current;
    }
  }
}

void unaryRelax(Scores scores, int nWords, int length, UnaryGrammar gr) {
  for(int start = 0; start <= nWords-length; start++) {
    int end = start + length;
    for(int i = 0; i < gr.size(); i++) {
      // Extract information from grammar rules
      tuple pair = gr[i].get(0)
      string symbol = pair.get(0)
      string lsym = pair.get(1)
      double rulescore = gr[i].get(1)
      double current = scores[start][end][symbol];
      for(int split = start+1; split <= end-1; split++) {
        double lscore = scores[start][split][lsym];
        double total = rulescore + lscore;
        if (total > current) {
          current = total;
        }
      }
      scores[start][end][symbol] = current;
    }
  }
}

Tree searchHighest (Scores scores){
  // TODO
}

Tree parse(vector<string> sen, Lexicons lex, BinaryGrammar gr2, UnaryGrammar gr1) {
  Scores scores = initScores();
  int nWords = sen.size();
  lexiconScores(scores, sen, nWords, lex);
  for(int spanlen = 2; i <= nWords; spanlen++) {
    binaryRelax(scores, nWords, spanlen, gr2);
    unaryRelax(scores, nWords, spanlen, gr1);
  }
  Tree result = searchHighest(scores);
  return result;
}

int main(){

}
