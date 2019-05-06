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
	WordToIdx wti;
	int num_symbol = read_symbols(sti, its);
	BinaryGrammar_SYM bg_s;
	UnaryGrammar_SYM ug_s;
	BinaryGrammar bg = read_binary_grammar(sti, bg_s);
	UnaryGrammar ug = read_unary_grammar(sti, ug_s);
	unordered_map<string, vector<tuple<string, vector<float>>>> lexicons = read_lexicon(sti, wti);
	vector<vector<string>> sentences = read_sentences();
	int* gr_b;
  int* gr_u;
  int* lens_b;
  int* lens_u;
  int* syms_b;
  int* syms_u;
  float* score_b;
  float* score_u;
  int num_blocks_b = generate_sym_to_rules_b(bg_s, gr_b, score_b, lens_b, syms_b);
  int num_blocks_u = generate_sym_to_rules_u(ug_s, gr_u, score_u, lens_u, syms_u);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end-start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	std::cout << "Total preprocessing time: " << elapsed_seconds.count() << "s\n";

	/* preprocessing done. Now need to start CUDA kernel */
	int num_sen = 40;
	start = std::chrono::system_clock::now();
	num_sen = 40;
	parseAllRuleBased(sentences, lexicons, bg, ug, num_symbol, sti, its, num_sen, wti);
	end = std::chrono::system_clock::now();

	elapsed_seconds = end-start;
	end_time = std::chrono::system_clock::to_time_t(end);
	std::cout << "Total parsing time (on CUDA, thread-based): " << elapsed_seconds.count() << "s\n";
	std::cout << "Average time per sentence (on CUDA, thread-based): " << elapsed_seconds.count()/num_sen << endl;

	int ub = (int)sentences.size();
	start = std::chrono::system_clock::now();
	int total = 0;
	int num = 0;
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
