#pragma once
#include <vector>
#include <memory>
#include "system.h"

namespace NEAT {
	// The class that provides the interface that users of the library must use for simulation.
	// Could morph into any system wishing to be given the desired behaviour through NEAT (eg flappy bird, inverted pendulem etc.).
	// Users of the library must inherit from this class in order to get the library to interface with their code.
	class Simulator {
	public:
		virtual void update_with_network_output(const std::vector<double>& net_outs) = 0;
		virtual const std::vector<double>& get_inputs_to_network() = 0;
		virtual double get_fitness() = 0;
		virtual void reset() = 0;
	};
}