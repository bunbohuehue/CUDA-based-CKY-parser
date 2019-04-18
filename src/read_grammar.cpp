#include "grammar.h"

static string grammarPath = "grammar.grammar";
static string lexiconPath = "grammar.lexicon";

vector<string> split(string line) {
    vector<string> res;
    int idx = 0;
    for (int i = 0; i < line.length(); i++){
        if (line[i] == ' ' || i == line.length() - 1) {
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
            completeRule = make_tuple(rule, stod(tmp[4]));
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
            rule = make_tuple(tmp[0], tmp[2]);
            completeRule = make_tuple(rule, stod(tmp[3]));
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
            scores.push_back(stod(s));
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

int main() {
    BinaryGrammar a = read_binary_grammar();
    read_unary_grammar();
    Lexicons l = read_lexicon();
    return 0;
}
