#pragma once
#include <cmath>
#include <memory>
#include <random>
#include <map>

#include "network.h"
#include "connection.h"
#include "simulator.h"

namespace NEAT {
	double modified_sigmoid(double input);
	double act_func(double input);
	double random(double thresh);
	uint32_t random_int(uint32_t ulim);

	// checks if v2 is a subset of v1
	template <typename T>
	bool is_subset(std::vector<T> v1, std::vector<T> v2)
	{
		std::sort(v1.begin(), v1.end());
		std::sort(v2.begin(), v2.end());
		return std::includes(v1.begin(), v1.end(), v2.begin(), v2.end());
	}

	class Network;

	class System {
	public:
		static std::default_random_engine rand_gen;
		static std::uniform_real_distribution<double> rand_dist;

		System(uint32_t size, uint32_t inputs, uint32_t outputs, double err);
		void init_simulators(const std::vector<std::shared_ptr<Simulator>>& sims);

		uint32_t get_innov_number(const Connection& gene);

		std::vector<Network>& get_population() { return population; }
		const std::vector<Network>& get_population() const { return population; }
		uint32_t get_size() const { return size; }
		const std::vector<uint32_t>& get_species_dist() const { return species_count; }

		void simulate_population(uint32_t timesteps);

		void speciate();
		void fitness_sharing();
		std::vector<uint32_t> assign_offspring();

	private:

		std::vector<std::shared_ptr<Simulator>> simulators; // the data passed to the population for simulation
		std::vector<Network> population;

		std::vector<Network> species_reps; // the species representatives for each species
		std::vector<uint32_t> species_count; // the overall number of organisms in each species

		std::vector<Connection> genes; // the current innovations of the population as a whole

		uint32_t inputs, outputs; // number of input nodes and output nodes including bias
		uint32_t size; // the overall population === population.size()

		// system initial conditions
		double spec_thresh; // delta_t

		double spec_c1;
		double spec_c2;
		double spec_c3;
	};

	template<typename Sim>
	void initialise_system(System& sys, Sim s) {
		std::vector<std::shared_ptr<Simulator>> simulators;
		for (uint32_t i = 0; i < sys.get_size(); ++i) {
			simulators.emplace_back(std::make_shared<Sim>(s));
		}

		sys.init_simulators(simulators);
	}

	// this version fills a vector for the user with shared pointers
	template<typename Sim>
	void initialise_system(System& sys, Sim s, std::vector<std::shared_ptr<Simulator>>& simulators) {
		for (uint32_t i = 0; i < sys.get_size(); ++i) {
			simulators.emplace_back(std::make_shared<Sim>(s));
		}

		sys.init_simulators(simulators);
	}
}