#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <stdint.h>
#include <algorithm>

#include "system.h"
#include "connection.h"
#include "simulator.h"

namespace NEAT {
	class System;

	double act_func(double);

	class Network {
	public:
		// constructor initialses all weights to 0.0
		Network(System& sys, uint32_t inputs, uint32_t outputs);

		// constructor initialses all weights randomly using std::uniform_real_distribution
		Network(System& sys, uint32_t inputs, uint32_t outputs, double random_thresh);

		// determines if the genome rhs is in the same species.
		// c1: coeficcient for disjoint genes
		// c2: coeficcient for excess genes
		// c3: coeficcient for weights
		// thresh is the speciation threshold - delta_t in the 2002 paper
		bool speciate(double c1, double c2, double c3, const std::vector<Connection>& rhs, double thresh);

		// the fitness before the adjustments according to explicit fitness sharing
		double get_raw_fitness() const { return fitness; }

		// adjust fitness does fitness sharing and divides the raw fitness by the number of individuals
		// in a species.
		void adjust_fitness(const System& sys);
		double get_shared_fitness() const { return shared_fitness; }

		// does what it says on the tin
		const std::vector<Connection>& get_genome() const { return genome; }
		uint32_t get_species() const { return species; }

		// calculates by propagating the activations through the network using act_func at each node
		const std::vector<double>& calculate(const std::vector<double>& inputs);

		// run the simulator for steps timesteps, and obtains fitness at the end of the run
		void simulate(std::shared_ptr<Simulator> sim, uint32_t steps);

		// performs crossover with rhs.
		// matching genes are inherited randomly
		// disjoint and excess genes are inherited from the fitter parent
		// disable_thresh: the probability that an offspring gene will be disabled
		// if it is disabled in either parent
		Network cross(const Network& rhs, double disable_thresh);
		static Network derive_from_genome(const std::vector<Connection>& genome, uint32_t, uint32_t);

		// parameters are probabilities of their respective types of mutations occuring
		void mutate(System& s, double node_mut, double conn_mut, double weight_mut, double mut_uniform, double err);

		class Node {
		public:
			Node(uint32_t node, uint8_t layer, const std::vector<uint32_t>& input_nodes)
				:node{ node }, layer{ layer }, input_nodes{ input_nodes }, value{}, back_inputs{} {}

			void calculate(double input) { value = act_func(input); }

			void update_back_inputs(const std::vector<Connection>& genome)
			{
				back_inputs.clear();

				for (uint32_t in_node : input_nodes)
				{
					auto in_conn = std::find(genome.begin(), genome.end(), Connection{ in_node, node });
					if (!in_conn->recursive && in_conn->enabled) {
						back_inputs.push_back(in_node);
					}
				}
			}

			void calculate_value(const std::vector<Connection>& genome)
			{
				double temp_val = 0;
				for (uint32_t n : input_nodes)
				{
					temp_val += std::find(genome.begin(), genome.end(), Connection{ n, node })->value;
				}

				calculate(temp_val);
			}

			double get_value() const { return value; }
			uint32_t get_layer() const { return layer; }
			uint32_t get_node() const { return node; }
			const std::vector<uint32_t>& get_inputs() const { return input_nodes; }
			const std::vector<uint32_t>& get_back_inputs() const { return back_inputs; }

			void set_layer(uint32_t new_layer) { layer = new_layer; }
			void set_value(double new_value) { value = new_value; }
			void add_input(uint32_t new_input) { input_nodes.push_back(new_input); }

			bool operator==(const Node& n) const { return n.get_node() == node; }

		private:
			double value; // value after activation function
			uint32_t node; // the index of the node in the Network's structure
			uint32_t layer;
			std::vector<uint32_t> input_nodes; // the nodes that the node is connected to for input
			std::vector<uint32_t> back_inputs; // the nodes that input non-recursive, enabled connections into the node
		};

	private:
		Network(uint32_t max_node, uint32_t inputs, uint32_t outputs)
			:fitness{ 0.0 }, shared_fitness{ 0.0 }, species{ 0 }, max_layer{ 0 }, inputs{ inputs }, outputs{ outputs }, node_num{ max_node } {}

		std::vector<Connection> genome;
		std::vector<Node> nodes;

		double fitness;
		double shared_fitness;
		uint32_t species;
		uint32_t inputs, outputs;
		uint32_t node_num; // one more than the maximum node number in the network
		uint32_t max_layer; // the layer number of the output nodes

		std::vector<double> output_data;

		// this procedure recalcaultes the layers of the nodes after a topology mutation
		// assumes layers are in vaild state (updated)
		void configure_layers();

		// mutate_uniform: the probability that a given weight will be mutated by adding to its original value
		// if not, it is assigned a new random value
		// err is the maximum value either side of zero
		void mutate_weights(double mutate_uniform, double err);

		// breaks a connection in two, disabling it.
		// connection in recieves a value of 1, connection out gets the old weight
		void mutate_add_node(System& sys);

		// attempts to connect 2 previously unconnected nodes
		void mutate_add_connection(System& sys, double err);

	};
}