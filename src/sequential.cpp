#include <iostream>
#include <tuple>
#include <vector>
#include <ctime>
#include "grammar.h"
using namespace std;

Scores initScores (int nWords, int num_symbol) {
	Scores scores(nWords + 1, vector<vector<double>>(nWords + 1, vector<double>(num_symbol, -DBL_MAX)));
	return scores;
}

void lexiconScores (Scores& scores, vector<string> sen, int nWords, unordered_map<string,
					vector<tuple<string, vector<double>>>> lex, SymToIdx sti, IdxToSym its, Occured& occured) {
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
				int tagidx = sti[subtag];
				scores[start][start+1][tagidx] = probs[j];
				occured[0][tagidx] = 1;
			}
		}
	}
}

void binaryRelax (Scores& scores, int nWords,
				  int length, BinaryGrammar gr, Occured& occured) {
	for(int i = 0; i < gr.size(); i++) {
		// Extract information from grammar rules
		tuple<int, int, int> pair = get<0>(gr[i]);
		int symbol = get<0>(pair);
		int lsym = get<1>(pair);
		int rsym = get<2>(pair);
		double rulescore = get<1>(gr[i]);
		for(int split = 1; split < length; split++) {
			if (occured[split-1][lsym] && occured[length-split-1][rsym]){
				for(int start = 0; start <= nWords-length; start++) {
					int end = start + length;
					double lscore = scores[start][start+split][lsym];
					if (lscore > -DBL_MAX) {
						double rscore = scores[start+split][end][rsym];
						if (rscore > -DBL_MAX) {
							double current = scores[start][end][symbol];
							double total = rulescore + lscore + rscore;
							if (total > current) {
								scores[start][end][symbol] = total;
								occured[length-1][symbol] = 1;
							}
						}
					}
				}
			}
		}
	}
}

void unaryRelax (Scores& scores, int nWords,
				 int length, UnaryGrammar gr, Occured& occured) {
	for(int i = 0; i < gr.size(); i++) {
		// Extract information from grammar rules
		tuple<int, int> pair = get<0>(gr[i]);
		int symbol = get<0>(pair);
		int lsym = get<1>(pair);
		if (occured[length-1][lsym]) {
			double rulescore = get<1>(gr[i]);
			for(int start = 0; start <= nWords-length; start++) {
				int end = start + length;
				double current = scores[start][end][symbol];
				if(scores[start][end][lsym] > -DBL_MAX) {
					double total = rulescore + scores[start][end][lsym];
					if (total > current) {
						scores[start][end][symbol] = total;
						occured[length-1][symbol] = 1;
					}
				}
			}
		}
	}
}

Ptree* searchHighest (Scores& scores, int symidx, vector<string> sen,
					  int start, int end, BinaryGrammar gr2, UnaryGrammar gr1, SymToIdx sti, IdxToSym its){
	Ptree* root = (Ptree*) malloc(sizeof(Ptree));
	if (symidx == -1) {
		double max = -DBL_MAX;
		int current = 0;
		for (int i = 0; i < scores[0][end].size(); i++) {
			if (scores[0][end][i] > max) {
				max = scores[0][end][i];
				current = i;
			}
		}
		root->symbol = its[current];
	} else {
		root->symbol = its[symidx];
	}
	Ptree* curr = root;
	for(int i = 0; i < gr1.size(); i++) {
		tuple<int, int> pair = get<0>(gr1[i]);
		int symbol = get<0>(pair);
		if (symbol == sti[curr->symbol]) {
			int lsym = get<1>(pair);
			double prob = get<1>(gr1[i]);
			if(scores[start][end][symbol] == scores[start][end][lsym] + prob){
				Ptree* child = (Ptree*) malloc(sizeof(Ptree));
				child->symbol = its[lsym];
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
		tuple<int, int, int> pair = get<0>(gr2[j]);
		int symbol = get<0>(pair);
		if (symbol == sti[curr->symbol]) {
			int lsym = get<1>(pair);
			int rsym = get<2>(pair);
			double rscore= get<1>(gr2[j]);
			for(int split = start+1; split <= end-1; split++) {
				if(scores[start][end][symbol] == scores[start][split][lsym]+scores[split][end][rsym]+rscore){
					curr->left = searchHighest(scores, lsym, sen, start, split, gr2, gr1, sti, its);
					curr->right = searchHighest(scores, rsym, sen, split, end, gr2, gr1, sti, its);
				}
			}
		}
	}
	return root;
}

Ptree* parse(vector<string> sen, unordered_map<string, vector<tuple<string, vector<double>>>> lex,
			 BinaryGrammar bg, UnaryGrammar ug, int num_symbol, SymToIdx sti, IdxToSym its) {
	int nWords = (int)sen.size();
	//std::cout << "total words: " << nWords << " \n";
	Scores scores = initScores(nWords, num_symbol);
	Occured occured(nWords, vector<bool>(num_symbol, 0));
	lexiconScores(scores, sen, nWords, lex, sti, its, occured);
	for(int spanlen = 2; spanlen <= nWords; spanlen++) {
		binaryRelax(scores, nWords, spanlen, bg, occured);
		unaryRelax(scores, nWords, spanlen, ug, occured);
	}
	Ptree* result = searchHighest(scores, -1, sen, 0, nWords, bg, ug, sti, its);
	return result;
}

int main(){
	auto start = std::chrono::system_clock::now();

	SymToIdx sti;
	IdxToSym its;
	int num_symbol = read_symbols(sti, its);
	BinaryGrammar bg = read_binary_grammar(sti);
	UnaryGrammar ug = read_unary_grammar(sti);
	unordered_map<string, vector<tuple<string, vector<double>>>> lexicons = read_lexicon(sti);
	vector<vector<string>> sentences = read_sentences();
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end-start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	std::cout << "Total preprocessing time: " << elapsed_seconds.count() << "s\n";

	/* START PARSING........... */
	int ub = (int)sentences.size();
	start = std::chrono::system_clock::now();
	int total = 0;
	int num = 0;
	int num_sen = 100;
	for (int i = 0; i < ub; i++){
		int len = (int)sentences[i].size();
		if (len <= 30 && num < num_sen) {
			num += 1;
			total += len;
			parse(sentences[i], lexicons, bg, ug, num_symbol, sti, its);
			cout << "Finished parsing sentence " << num << endl;
		}
	}
	std::cout << "avg len: " << total/num_sen << " \n";
	end = std::chrono::system_clock::now();

	elapsed_seconds = end-start;
	end_time = std::chrono::system_clock::to_time_t(end);
	std::cout << "Total parsing time: " << elapsed_seconds.count() << "s\n";
	std::cout << "Average time per sentence" << elapsed_seconds.count()/num_sen << endl;
}
