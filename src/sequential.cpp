#include <iostream>
#include <tuple>
#include <vector>
#include "grammar.h"
using namespace std;

Scores initScores(int nWords, unordered_map<string, double> symbols) {
  Scores scores(nWords, vector<unordered_map<string, double>>(nWords, symbols));
  return scores;
}

void lexiconScores(Scores scores, vector<string> sen, int nWords,
  unordered_map<string, vector<tuple<string, vector<double>>>> lex) {
  for(int start = 0; start < nWords; start++) {
    string word = sen[start];
    vector<tuple<string, vector<double>>> rules = lex[word];
    for(int i = 0; i < rules.size(); i++) {
      // Extract information from grammar rules
      tuple pair = rules[i];
      string tag = pair.get(0);
      vector<double> probs = pair.get(1);
      for(int j = 0; j < probs.size(); j++) {
        string subtag = tag + '_' + to_string(j);
        scores[start][start+1][subtag] = probs[j];
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
        double rscore = scores[split][end][rsym];
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
      double total = rulescore + scores[start][end][lsym];
      if (total > current) {
        scores[start][end][symbol] = total;
      }
    }
  }
}

Ptree searchHighest (Scores scores, str symbol, string sen, int start, int end, BinaryGrammar gr2, UnaryGrammar gr1){
  /* Root = Node;
    if symbol == "ATROOT"
      Root.symbol = argmax(scores[0][nWords]);
    else
      Root.symbol = symbol
    curr = Root;
     for rule in gr1:
        rule: symbol -> lsym
        if symbol == curr.symbol
          if socres[start][end][Root.symbol] + Prule == socres[start][end][lsym]
              child = Node lsym;
              Root.leftchild = child
              curr = child
    if start +1 == end:
      curr.child = Node
      Node.symbol = sen(start)
      return Root
    for rule in gr2:
      rule : A ->  B C
      if A == curr.symbol:
        for (split = start+1, split <= end-1; split ++)
          if (socres[start][end][curr.symbol] == rulescore + score[start][split][B] + score[split][end][C])
            curr.left = searchHighest(scores, B, start, split, gr2, gr1)
            curr.right = searchHighest(scores, C, split,end,gr2,gr1)
     return Root;
  */
}

Ptree parse(vector<string> sen, unordered_map<string, vector<tuple<string, vector<double>>>> lex,
   BinaryGrammar gr2, UnaryGrammar gr1, unordered_map<string, double> symbols) {
  int nWords = sen.size();
  Scores scores = initScores(nWords, symbols);
  lexiconScores(scores, sen, nWords, lex);
  for(int spanlen = 2; i <= nWords; spanlen++) {
    binaryRelax(scores, nWords, spanlen, gr2);
    unaryRelax(scores, nWords, spanlen, gr1);
  }
  Ptree result = searchHighest(scores, "ATROOT", sen, 0, nWords, gr2, gr1);
  return result;
}

int main(){
  BinaryGrammar bg = read_binary_grammar();
  UnaryGrammar ug = read_unary_grammar();
  unordered_map<string, vector<tuple<string, vector<double>>>> lexicons = read_lexicon();
  /*
  unordered_map<string, double> symbols = ?
  for sen in stentences
    ptree = parse(sen, lexicons, gr2, gr1, symbols);

  */
}
