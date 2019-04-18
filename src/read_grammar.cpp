#include "grammar.h"

static string grammarPath = "../grammar/grammar.grammar";
static string lexiconPath = "../grammar/grammar.lexicon";

vector<string> split(string line) {
  vector<string> res;
  int idx = 0;
  for (int i = 0; i < line.length(); i++){
    if (line[i] == ' ') {
      int len = i - idx;
      res.push_back(line.substr(idx, i));
    }
  }
  return res;
}

BinaryGrammar read_binary_grammar() {
  BinaryGrammar result;
  string line;
  vector<string> tmp;
  tuple<string, string, string> rule;
  tuple<tuple<string, string, string>, double> completeRule;

  ifstream grammarfile(grammarPath);
  if (!grammarfile) {
    cout << "Error opening file." << endl;
    return result;
  }
  while (getline(grammarfile, line)) {
    // parse the line and put it into the grammar
    tmp = split(line);
    if (res.size() == 5) {
      rule = make_tuple(tmp[0], tmp[2], tmp[3]);
      completeRule = make_tuple(rule, stof(tmp[4]));
      result.push_back(completeRule);
    }
  }
  return result;
}

UnaryGrammar read_unary_grammar() {
  UnaryGrammar result;
  string line;
  vector<string> tmp;
  tuple<string, string> rule;
  tuple<tuple<string, string>, double> completeRule;

  ifstream grammarfile(grammarPath);
  if (!grammarfile) {
    cout << "Error opening file." << endl;
    return result;
  }
  while (getline(grammarfile, line)) {
    // parse the line and put it into the grammar
    tmp = split(line);
    if (res.size() == 4) {
      rule = make_tuple(tmp[0], tmp[2]);
      completeRule = make_tuple(rule, stof(tmp[3]));
      result.push_back(completeRule);
    }
  }
  return result;
}

Lexicons read_lexicon() {
  Lexicons result;
  string line;
  vector<string> tmp;
  vector<double> scores;
  tuple<string, string, vector<double>> lexicon;

  ifstream grammarfile(lexionPath);
  if (!grammarfile) {
    cout << "Error opening file." << endl;
    return result;
  }
  while (getline(grammarfile, line)) {
    // parse the line and put it into the grammar
    tmp = split(line);
    for (int i = 2; i < tmp.size(); i++) {
      string s = tmp[i];
      if (i == 2) s = s.substr(1, s.length() - 1);
      else if (i == tmp.size() - 1) s = s.substr(0, s.length() - 1);
      scores.push_back(stof(s));
    }
    lexicon = make_tuple(tmp[1], tmp[0], scores);
    result.push_back(lexicon);
    scores.clear();
  }

  return result;
}
