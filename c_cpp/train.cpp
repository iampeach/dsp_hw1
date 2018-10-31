#include <iostream>
#include <vector>
#include <fstream>
#include "hmm.h"

using namespace std;

// static global variables
static double _alpha[MAX_SEQ][MAX_STATE];
static double _beta[MAX_SEQ][MAX_STATE];
static double _gamma[MAX_SEQ][MAX_STATE];
static double _initial[MAX_STATE];
static double _gamma_sum_to_T_minus_1[MAX_STATE];
static double _gamma_sum_to_T[MAX_STATE];
static double _observ_gamma[MAX_OBSERV][MAX_STATE];
static double _epsilon[MAX_SEQ][MAX_STATE][MAX_STATE];
static double _epsilon_sum[MAX_STATE][MAX_STATE];

static HMM hmm_model;
static vector<string> input_data;

// function prototypes
bool load_data(string);
void initialize();
void train(vector<string>&);
void forward(string&);
void backward(string&);
void set_gamma(string&);
void set_epsilon(string&);
void update_model();

int main(int argc, char* argv[]) {
	// TODO Error detect
	int iterations = strtol(argv[1], NULL, 10);
	char* init_model = argv[2];
	string input_file = argv[3];
	char* output_file = argv[4];

	loadHMM( &hmm_model, init_model );
	
	if ( !load_data(input_file) )
		return -1;

	for ( size_t iter = 0; iter < iterations; ++iter) {
		initialize();
		train(input_data);
		update_model();
	}
	
	FILE *output_model = fopen(output_file, "w");
	dumpHMM(output_model, &hmm_model);
}

bool load_data(string data_src_name) {
	ifstream data_src(data_src_name);

	if ( !data_src ) {
		cout << "Input File Not Found" << endl;
		return false;
	}
	
	string input_buffer;
	while ( !data_src.eof() ) {
		getline(data_src, input_buffer);
		if ( !input_buffer.empty() )
			input_data.push_back(input_buffer);
	}

	data_src.close();
	return true;
}

void initialize() {
	memset(_initial, .0, sizeof _initial);
	memset(_gamma_sum_to_T_minus_1, .0, sizeof _gamma_sum_to_T_minus_1);
	memset(_gamma_sum_to_T, .0, sizeof _gamma_sum_to_T);
	memset(_observ_gamma, .0, sizeof _observ_gamma);
	memset(_epsilon_sum, .0, sizeof _epsilon_sum);
}

void train(vector<string> &input_data) {
	for ( size_t idx = 0; idx < input_data.size(); ++idx ) {
		forward(input_data[idx]);
		backward(input_data[idx]);
		set_gamma(input_data[idx]);
		set_epsilon(input_data[idx]);
	}
}

void forward(string &observation) {
	for ( size_t j = 0; j < hmm_model.state_num; ++j )
		_alpha[0][j] = hmm_model.initial[j] * hmm_model.observation[observation[0]-'A'][j];
	for ( size_t t = 1; t < observation.size(); ++t ) {
		for ( size_t j = 0; j < hmm_model.state_num; ++j ) {
			for ( size_t i = 0; i < hmm_model.state_num; ++i )
				_alpha[t][j] += _alpha[t-1][i] * hmm_model.transition[i][j];
			_alpha[t][j] *= hmm_model.observation[observation[t]-'A'][j];
		}
	}
}

void backward(string &observation) {
	for ( size_t j = 0; j < hmm_model.state_num; ++j )
		_beta[observation.size()-1][j] = 1;
	for ( int t = observation.size()-2; t >= 0; --t )
		for ( size_t i = 0; i < hmm_model.state_num; ++i )
			for ( size_t j = 0; j < hmm_model.state_num; ++j ) 
				_beta[t][i] += _beta[t+1][j] * hmm_model.transition[i][j] * hmm_model.observation[observation[t+1]-'A'][j];
}

void set_gamma(string &observation) {
	for ( size_t t = 0; t < observation.size(); ++t ) {
		double summation = .0;
		for ( size_t i = 0; i < hmm_model.state_num; ++i )
			summation += _alpha[t][i] * _beta[t][i];

		for ( size_t i = 0; i < hmm_model.state_num; ++i ) {
			_gamma[t][i] = _alpha[t][i] * _beta[t][i] / summation;
			if ( t < observation.size()-1 )
				_gamma_sum_to_T_minus_1[i] += _gamma[t][i];
			_gamma_sum_to_T[i] += _gamma[t][i];
			_observ_gamma[observation[t]-'A'][i] += _gamma[t][i];
		}
	}
	for ( size_t i = 0; i < hmm_model.state_num; ++i) {
		_initial[i] += _gamma[0][i];
	}
}

void set_epsilon(string &observation) {
	for ( size_t t = 0; t < observation.size()-1; ++t ) {
		double summation = .0;
		for ( size_t i = 0; i < hmm_model.state_num; ++i )
			for ( size_t j = 0; j < hmm_model.state_num; ++j )
				summation += _alpha[t][i] * hmm_model.transition[i][j] * hmm_model.observation[observation[t+1]-'A'][j] * _beta[t+1][j];

		for ( size_t i = 0; i < hmm_model.state_num; ++i ) {
			for ( size_t j = 0; j < hmm_model.state_num; ++j ) {
				_epsilon[t][i][j] = _alpha[t][i] * hmm_model.transition[i][j] * hmm_model.observation[observation[t+1]-'A'][j] * _beta[t+1][j] / summation;
				_epsilon_sum[i][j] += _epsilon[t][i][j];
			}
		}
	}
}

void update_model() {
	for ( size_t i = 0; i < hmm_model.state_num; ++i )
		hmm_model.initial[i] = _initial[i] / input_data.size();
	
	for ( size_t i = 0; i < hmm_model.state_num; ++i )
		for ( size_t j = 0; j < hmm_model.state_num; ++j )
			hmm_model.transition[i][j] = _epsilon_sum[i][j] / _gamma_sum_to_T_minus_1[i];

	for ( size_t k = 0; k < hmm_model.observ_num; ++k )
		for ( size_t j = 0; j < hmm_model.state_num; ++j )
			hmm_model.observation[k][j] = _observ_gamma[k][j] / _gamma_sum_to_T[j];
}
