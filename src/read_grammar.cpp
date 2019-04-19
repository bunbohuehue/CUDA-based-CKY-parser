#include "grammar.h"
#include <math.h>

static string grammarPath = "grammar.grammar";
static string lexiconPath = "grammar.lexicon";
static string sentencesPath = "ptb.2-21.short.txt";

vector<string> split(string line) {
  vector<string> res;
  int idx = 0;
  for (int i = 0; i <= line.length(); i++){
    if (line[i] == ' ' || i == line.length()) {
      int len = i - idx;
      res.push_back(line.substr(idx, len));
      idx = i + 1;
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

  ifstream grammarfile("grammar.grammar", ios::in);
  if (!grammarfile.is_open()) {
    cout << "Error opening file." << endl;
    return result;
  }
  while (getline(grammarfile, line)) {
    // parse the line and put it into the grammar
    tmp = split(line);
    if (tmp.size() == 5) {
      rule = make_tuple(tmp[0], tmp[2], tmp[3]);
      completeRule = make_tuple(rule, log(stod(tmp[4])));
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
    if (tmp.size() == 4) {
      if (tmp[0] == tmp[2] && stod(tmp[3]) == 1.0) continue;
      rule = make_tuple(tmp[0], tmp[2]);
      completeRule = make_tuple(rule, log(stod(tmp[3])));
      result.push_back(completeRule);
    }
  }
  return result;
}

unordered_map<string, vector<tuple<string, vector<double>>>> read_lexicon() {
  unordered_map<string, vector<tuple<string, vector<double>>>> result;
  string line;
  vector<string> tmp;
  vector<double> scores;
  tuple<string, vector<double>> tag;

  ifstream grammarfile(lexiconPath);
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
      scores.push_back(log(stod(s)));
    }
    tag = make_tuple(tmp[0], scores);
    auto it = result.find(tmp[1]);
    if (it == result.end()) {
      // key is not there
      vector<tuple<string, vector<double>>> lexicon;
      lexicon.push_back(tag);
      result.insert(pair<string, vector<tuple<string, vector<double>>>>(tmp[1], lexicon));
    }
    else {
      it->second.push_back(tag);
    }
    scores.clear();
  }
  return result;
}

vector<vector<string>> read_sentences() {
  vector<vector<string>> sentences;
  vector<string> tmp;
  ifstream senfile(sentencesPath);
  string line;

  if (!senfile) {
    cout << "Error opening file." << endl;
    return sentences;
  }
  while (getline(senfile, line)) {
    tmp = split(line);
    sentences.push_back(tmp);
  }
  return sentences;
}

unordered_map<string, double> read_symbols(){
  unordered_map<string, double> result;
  vector<string> tmp;
  string line;

  ifstream grammarfile(grammarPath);
  if (!grammarfile) {
    cout << "Error opening file." << endl;
    return result;
  }
  while (getline(grammarfile, line)) {
    // parse the line and put it into the grammar
    tmp = split(line);
    for (int i = 0; i < 4; i++) {
      if (i == 1) continue;
      if (i < 3 || (i == 3 && tmp.size() == 5)) {
        auto it = result.find(tmp[i]);
        if (it == result.end()) {
          result.insert(pair<string, double>(tmp[i], -DBL_MAX));
        }
      }
    }
  }
  return result;
}

/*int main() {
BinaryGrammar a = read_binary_grammar();
read_unary_grammar();
Lexicons l = read_lexicon();
vector<vector<string>> sentences = read_sentences();
unordered_map<string, double> symbols = read_symbols();
return 0;
}*/
