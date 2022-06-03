#pragma once
#include <cmath>
#include <memory>
#include <random>
#include <algorithm>
#include <numeric>
#include <thread>

#include "network.h"
#include "connection.h"
#include "simulator.h"

namespace NEAT {
	double modified_sigmoid(double input);
	double act_func(double input);
	double random(double thresh);
	uint32_t random_int(uint32_t ulim); // inclusive

	// checks if v2 is a subset of v1
	template <typename T>
	bool is_subset(std::vector<T> v1, std::vector<T> v2)
	{
		std::sort(v1.begin(), v1.end());
		std::sort(v2.begin(), v2.end());
		return std::includes(v1.begin(), v1.end(), v2.begin(), v2.end());
	}

	class Network;
	class Simulator;

	struct Species {
		Species(const Network& net);

		bool progress_made(uint32_t since);
		void set_rep(const Network& net);
		const Network& get_rep() const { return *rep; }

		uint32_t count; // how many organisms in species
		uint32_t offspring; // how many organisms in species
		std::vector<double> fitness_log;

	private:
		std::unique_ptr<Network> rep; // genome that represents species
	};

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
		const std::vector<Species>& get_species() const { return species; }

		void simulate_population(uint32_t timesteps);
		void simulate_multithread(uint32_t timesteps);
		void reset_simulators();

		void produce_next_generation();

		std::ostream& log(std::ostream&);
		double get_max_fitness() const { return max_fitness; }
		double get_mean_fitness() const { return mean_fitness; }
		uint32_t get_generation() const { return generation; }

		std::ostream& dump_fittest(std::ostream&);

	private:

		std::vector<std::shared_ptr<Simulator>> simulators; // the data passed to the population for simulation
		std::vector<Network> population;

		//std::vector<Network> species_reps; // the species representatives for each species
		//std::vector<uint32_t> species_count; // the overall number of organisms in each species
		//std::vector<std::vector<double>> fitness_trends; // uesd to decide whether to eliminate a species
		std::vector<Species> species; // to replace the preceding 3 lines

		std::vector<Connection> genes; // the current innovations of the population as a whole

		uint32_t inputs, outputs; // number of input nodes and output nodes including bias
		uint32_t size; // the overall population === population.size()

		uint32_t generation;

		// system initial conditions
		double spec_thresh; // delta_t
		uint32_t target_species;
		uint32_t stagnation_gen; // how many generations of stagnation forces a species to be removed

		double spec_c1;
		double spec_c2;
		double spec_c3;

		// crossover: probability that a gene is disabled if it is disabled in either parent
		double disable_thresh;

		double keep; // the percentage of the genomes to reproduce from
		double crossover_rate; // fraction of offspring that result from crossover
		double spec_penalty;

		// mutation probabilities
		double node_mut;
		double conn_mut;
		double weight_mut;
		double mut_uniform;

		double weight_err; // the error to mutate (+-weight_err)

		double mean_fitness, mean_hidden_nodes, max_fitness;

		void speciate();
		void update_reps(); // assumes speciated population
		void fitness_sharing(); // carry out the fitness sharing algorithm (pg. 110)
		void update_fitness_log(); // assumes freshly speciated population

		// assumes fitness shared and speciated population
		void assign_offspring(); // give the number of offspring to each species

		// assumes speciated population
		void cull_population(); // removes the unfit genomes from the population
		static void simulate_subset(System* s, uint32_t first, uint32_t last, uint32_t steps); // for multithreading
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