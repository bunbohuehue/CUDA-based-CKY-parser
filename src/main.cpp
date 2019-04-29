#include <iostream>
#include <tuple>
#include <vector>
#include <ctime>
#include <chrono>
#include "grammar.h"
#include "cuda_parser.h"
using namespace std;

int main(){
	auto start = std::chrono::system_clock::now();

	SymToIdx sti;
	IdxToSym its;
  Symbols syms;
	int num_symbol = read_symbols(sti, its);
	BinaryGrammar bg = read_binary_grammar(sti);
	UnaryGrammar ug = read_unary_grammar(sti);
	unordered_map<string, vector<tuple<string, vector<float>>>> lexicons = read_lexicon(sti);
	vector<vector<string>> sentences = read_sentences();
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end-start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	std::cout << "Total preprocessing time: " << elapsed_seconds.count() << "s\n";

	/* preprocessing done. Now need to start CUDA kernel */
	int ub = (int)sentences.size();
	start = std::chrono::system_clock::now();
	int total = 0;
	int num = 0;
	int num_sen = 40;
	for (int i = 0; i < ub; i++){
		int len = (int)sentences[i].size();
		if (num < num_sen) {
			num += 1;
			total += len;
			parse(sentences[i], lexicons, bg, ug, num_symbol, sti, its);
			cout << "Finished parsing sentence (CUDA) " << num << endl;
		}
	}
	std::cout << "avg len: " << total/num_sen << " \n";
	end = std::chrono::system_clock::now();

	elapsed_seconds = end-start;
	end_time = std::chrono::system_clock::to_time_t(end);
	std::cout << "Total parsing time (on CUDA): " << elapsed_seconds.count() << "s\n";
	std::cout << "Average time per sentence (on CUDA): " << elapsed_seconds.count()/num_sen << endl;

	start = std::chrono::system_clock::now();
	total = 0;
	num = 0;
	for (int i = 0; i < ub; i++){
		int len = (int)sentences[i].size();
		if (num < num_sen) {
			num += 1;
			total += len;
			parse_sequential(sentences[i], lexicons, bg, ug, num_symbol, sti, its);
			cout << "Finished parsing sentence (CPU) " << num << endl;
		}
	}
	std::cout << "avg len: " << total/num_sen << " \n";
	end = std::chrono::system_clock::now();

	elapsed_seconds = end-start;
	end_time = std::chrono::system_clock::to_time_t(end);
	std::cout << "Total parsing time (on CPU): " << elapsed_seconds.count() << "s\n";
	std::cout << "Average time per sentence (on CPU): " << elapsed_seconds.count()/num_sen << endl;
}
