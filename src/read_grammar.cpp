#include "grammar.h"
#include <math.h>

static string grammarPath = "grammar.grammar";
static string lexiconPath = "grammar.lexicon";
static string sentencesPath = "ptb.2-21.txt";

vector<string> split(string line) {
  vector<string> res;
  int idx = 0;
  for (int i = 0; i <= line.length(); i++){
    if (line[i] == ' ' || i == line.length()) {
      int len = i - idx;
      if(len > 0) {
        res.push_back(line.substr(idx, len));
        idx = i + 1;
      }
    }
  }
  return res;
}

BinaryGrammar read_binary_grammar(SymToIdx sti, BinaryGrammar_SYM& bg) {
  BinaryGrammar result;
  string line;
  vector<string> tmp;
  tuple<int, int, int> rule;
  tuple<tuple<int, int, int>, float> completeRule;

  ifstream grammarfile("grammar.grammar", ios::in);
  if (!grammarfile.is_open()) {
    cout << "Error opening file." << endl;
    return result;
  }
  while (getline(grammarfile, line)) {
    // parse the line and put it into the grammar
    tmp = split(line);
    if (tmp.size() == 5) {
      int head = sti[tmp[0]];
      rule = make_tuple(sti[tmp[0]], sti[tmp[2]], sti[tmp[3]]);
      completeRule = make_tuple(rule, log(stod(tmp[4])));
      result.push_back(completeRule);

      auto it = bg.find(head);
      if (it == bg.end()) {
        vector<tuple<int, int, float>> t;
        t.push_back(make_tuple(sti[tmp[2]], sti[tmp[3]], log(stod(tmp[4]))));
        bg.insert(pair<int, vector<tuple<int, int, float>>>(head, t));
      }
      else {
        bg[head].push_back(make_tuple(sti[tmp[2]], sti[tmp[3]], log(stod(tmp[4]))));
      }
    }
  }
  return result;
}

UnaryGrammar read_unary_grammar(SymToIdx sti, UnaryGrammar_SYM& ug) {
  UnaryGrammar result;
  string line;
  vector<string> tmp;
  tuple<int, int> rule;
  tuple<tuple<int, int>, float> completeRule;

  ifstream grammarfile(grammarPath);
  if (!grammarfile) {
    cout << "Error opening file." << endl;
    return result;
  }
  while (getline(grammarfile, line)) {
    // parse the line and put it into the grammar
    tmp = split(line);
    if (tmp.size() == 4) {
      int head = sti[tmp[0]];
      if (tmp[0] == tmp[2] && stod(tmp[3]) == 1.0) continue;
      rule = make_tuple(sti[tmp[0]], sti[tmp[2]]);
      completeRule = make_tuple(rule, log(stod(tmp[3])));
      result.push_back(completeRule);

      auto it = ug.find(head);
      if (it == ug.end()) {
        vector<tuple<int, float>> t;
        t.push_back(make_tuple(sti[tmp[2]], log(stod(tmp[3]))));
        ug.insert(pair<int, vector<tuple<int, float>>>(head, t));
      }
      else {
        ug[head].push_back(make_tuple(sti[tmp[2]], log(stod(tmp[3]))));
      }
    }
  }
  return result;
}

unordered_map<string, vector<tuple<string, vector<float>>>> read_lexicon(SymToIdx sti) {
  unordered_map<string, vector<tuple<string, vector<float>>>> result;
  string line;
  vector<string> tmp;
  vector<float> scores;
  tuple<string, vector<float>> tag;

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
      vector<tuple<string, vector<float>>> lexicon;
      lexicon.push_back(tag);
      result.insert(pair<string, vector<tuple<string, vector<float>>>>(tmp[1], lexicon));
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

int read_symbols(SymToIdx& sti, IdxToSym& its){
  unordered_map<string, float> result;
  vector<string> tmp;
  string line;

  ifstream grammarfile(grammarPath);
  if (!grammarfile) {
    cout << "Error opening file." << endl;
    return 0;
  }

  int num_symbol = 0;
  while (getline(grammarfile, line)) {
    // parse the line and put it into the grammar
    tmp = split(line);
    for (int i = 0; i < 4; i++) {
      if (i == 1) continue;
      if (i < 3 || (i == 3 && tmp.size() == 5)) {
        auto it = result.find(tmp[i]);
        if (it == result.end()) {
          // we have found a brand new symbol
          result.insert(pair<string, float>(tmp[i], -DBL_MAX));
          // also, update sti and its
          sti.insert(pair<string, int>(tmp[i], num_symbol));
          its.insert(pair<int, string>(num_symbol, tmp[i]));
          num_symbol++;
        }
      }
    }
  }
  return num_symbol;
}

int generate_sym_to_rules_b(BinaryGrammar_SYM bg, int** rule_arr, float** score_arr, int** lens, int** syms) {
  int num_blocks = 0;
  for (auto it : bg) {
    num_blocks += (it.second.size() /1024) + 1;
  }
  *rule_arr = (int*) malloc(num_blocks * 1024 * 2 * sizeof(int));
  *score_arr = (float*) malloc(num_blocks * 1024 * sizeof(float));
  *lens = (int*) malloc(num_blocks * sizeof(int));
  *syms = (int*) malloc(num_blocks * sizeof(int));

  for (auto it : bg) {
    int symbol = it.first;
    vector<tuple<int, int, float>> rules = it.second;
    int num_rows = (rules.size() / 1024) + 1;
    for (int i = 0; i < num_rows; i++) {
      *syms[i] = symbol;
      *lens[i] = (rules.size() - (i + 1) * 1024) >= 0 ? 1024 : (rules.size() - i * 1024);
      for (int j = 0; j < *lens[i]; j++) {
        tuple<int, int, float> t = rules[i * 1024 + j];
        *rule_arr[i * 2048 + 2 * j] = get<0>(t);
        *rule_arr[i * 2048 + 2 * j + 1] = get<1>(t);
        *score_arr[i * 1024 + j] = get<2>(t);
      }
    }
  }
  return num_blocks;
}

int generate_sym_to_rules_u(UnaryGrammar_SYM ug, int** rule_arr, float** score_arr, int** lens, int** syms) {
  int num_blocks = 0;
  for (auto it : ug) {
    num_blocks += (it.second.size() /1024) + 1;
  }
  *rule_arr = (int*) malloc(num_blocks * 1024 * sizeof(int));
  *score_arr = (float*) malloc(num_blocks * 1024 * sizeof(float));
  *lens = (int*) malloc(num_blocks * sizeof(int));
  *syms = (int*) malloc(num_blocks * sizeof(int));

  for (auto it : ug) {
    int symbol = it.first;
    vector<tuple<int, float>> rules = it.second;
    int num_rows = (rules.size() / 1024) + 1;
    for (int i = 0; i < num_rows; i++) {
      *syms[i] = symbol;
      *lens[i] = (rules.size() - (i + 1) * 1024) >= 0 ? 1024 : (rules.size() - i * 1024);
      for (int j = 0; j < *lens[i]; j++) {
        tuple<int, float> t = rules[i * 1024 + j];
        *rule_arr[i * 1024 + j] = get<0>(t);
        *score_arr[i * 1024 + j] = get<1>(t);
      }
    }
  }
  return num_blocks;
}
