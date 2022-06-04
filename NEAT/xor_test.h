#pragma once
#include <cmath>

#include "simulator.h"

class XOR : public NEAT::Simulator {
public:
	XOR() :xor_input(2), expected_output{ 0 }, fitness{ 0 }, test_count{ 0 } {}

	void update_with_network_output(const std::vector<double>& net_outs) override {
		double output = net_outs[0];
		fitness += 1 - pow((expected_output - output), 2);
	}

	const std::vector<double>& get_inputs_to_network() override {
		switch (test_count) {
		case 0:
			xor_input = { 0, 0 };
			expected_output = 0;
			break;

		case 1:
			xor_input = { 0, 1 };
			expected_output = 1;
			break;

		case 2:
			xor_input = { 1, 0 };
			expected_output = 1;
			break;

		case 3:
			xor_input = { 1, 1 };
			expected_output = 0;
			break;

		default:
			break;
		}

		if (test_count < 3) test_count++;
		else test_count = 0;

		return xor_input;
	}

	double get_fitness() override { 
		//if (fitness == 4) throw std::runtime_error("Solution found.");
		return fitness;
	}
	void reset() override { test_count = 0; fitness = 0; }

private:
	std::vector<double> xor_input;
	uint8_t test_count;
	double expected_output;
	double fitness;
};