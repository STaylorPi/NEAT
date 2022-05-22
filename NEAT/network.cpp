#include "network.h"

namespace NEAT {
	Network::Network(System& sys, uint32_t inputs, uint32_t outputs)
		:inputs{ inputs }, outputs{ outputs }, fitness{}, species{}, nodes{}, output_data(outputs), max_layer{ 0 }
	{
		for (uint32_t inn = 0; inn < inputs; ++inn) {
			for (uint32_t outn = 0; outn < outputs; ++outn) {
				// constructor automatically initialises to weight 0.0
				uint32_t innov = genome.size();
				genome.emplace_back(Connection{ inn, inputs + outn, true, 0, false });
				genome[genome.size() - 1].innov_num = sys.get_innov_number(genome[genome.size() - 1]);
			}
		}

		for (uint32_t n = 0; n < inputs; ++n) {
			nodes.emplace_back(Node{ n, 0, {} });
		}

		std::vector<uint32_t> in_nodes(inputs); // the default inputs for the output nodes initialised below
		for (uint32_t i = 0; i < inputs; ++i) in_nodes[i] = i;

		for (uint32_t n = 0; n < outputs; ++n) {
			nodes.emplace_back(Node{ n + inputs, 1, in_nodes });
		}

		node_num = inputs + outputs;
	}

	Network::Network(System& sys, uint32_t inputs, uint32_t outputs, double random_thresh)
		:inputs{ inputs }, outputs{ outputs }, fitness{}, species{}, nodes{}, output_data(outputs), max_layer{ 0 }
	{
		for (uint32_t inn = 0; inn < inputs; ++inn) {
			for (uint32_t outn = 0; outn < outputs; ++outn) {
				uint32_t innov = genome.size();
				genome.emplace_back(Connection{ inn, inputs + outn, true, (System::rand_dist(System::rand_gen) * random_thresh * 2) - random_thresh, 0, false});
				genome[genome.size() - 1].innov_num = sys.get_innov_number(genome[genome.size() - 1]);
			}
		}

		for (uint32_t n = 0; n < inputs; ++n) {
			nodes.emplace_back(Node{ n, 0, {} });
		}

		std::vector<uint32_t> in_nodes(inputs); // the default inputs for the output nodes initialised below
		for (uint32_t i = 0; i < inputs; ++i) in_nodes[i] = i;

		for (uint32_t n = 0; n < outputs; ++n) {
			nodes.emplace_back(Node{ n + inputs, 1, in_nodes });
		}

		node_num = inputs + outputs;
	}

	bool Network::speciate(double c1, double c2, double c3, const std::vector<Connection>& rhs, double thresh)
	{
		uint32_t max_innov_this = std::max_element(genome.begin(), genome.end()) -> innov_num;
		uint32_t max_innov_rhs = std::max_element(rhs.begin(), rhs.end()) -> innov_num;

		double weight_diff_sum = 0; // the sum of the weights
		uint32_t match = 0;
		uint32_t disjoint = 0;
		uint32_t excess = 0;

		for (uint32_t innov = 0; innov <= std::max(max_innov_this, max_innov_rhs); ++innov)
		{
			auto rhs_conn_iter = std::find_if(rhs.begin(), rhs.end(), [&](const Connection& carg) {return (carg.innov_num == innov); });
			auto this_conn_iter = std::find_if(genome.begin(), genome.end(), [&](const Connection& carg) {return (carg.innov_num == innov); });
			if (rhs_conn_iter != rhs.end() && this_conn_iter != genome.end()) // the gene is in both genomes
			{
				match++;
				weight_diff_sum += std::abs(rhs_conn_iter->weight - this_conn_iter->weight);
			}
			else if (rhs_conn_iter == rhs.end() && this_conn_iter != genome.end()) // the gene is in this->genome but not rhs
			{
				if (this_conn_iter->innov_num < max_innov_rhs) disjoint++; // the gene in this->genome is within range of rhs
				else excess++;
			}
			else if (rhs_conn_iter != rhs.end() && this_conn_iter == genome.end()) // the gene is in rhs but not in this->genome
			{
				if (rhs_conn_iter->innov_num < max_innov_this) disjoint++; // the gene in rhs->genome is within range of rhs
				else excess++;
			}
		}

		uint32_t max_size = std::max(genome.size(), rhs.size());
		std::cout << weight_diff_sum / match << '\n';
		double delta = c1 * (disjoint / double(max_size)) + c2 * (excess / double(max_size)) + c3 * (weight_diff_sum / match);

		return (delta <= thresh);
	}

	const std::vector<double>& Network::calculate(const std::vector<double>& input_data)
	{
		// sanity check
		if (input_data.size() != inputs) throw std::runtime_error("Incorrect input array size to NEAT::Network::calculate");

		uint32_t working_layer = 0;

		// initialise the neurons in layer zero (input layer)
		// NB: caller is responsible for bias input
		for (Node& n : nodes) {
			if (n.get_layer() == 0) n.set_value(input_data[n.get_node()]);
		}

		// propagate forwards through the layers
		while (working_layer <= max_layer) {
			// update the connections that have inputs on the working layer
			for (const Node& n : nodes) {
				if (n.get_layer() == working_layer) {
					for (Connection& c : genome) {
						if (c.node1 == n.get_node()) c.update_value(n.get_value());
					}
				}
			}

			// update neurons in the next layer
			working_layer++;
			// the previous for-loop only updates recursive connections on the last iteration - no more nodes to compute
			if (working_layer <= max_layer) {
				for (Node& n : nodes)
				{
					// since all the connections feeding in are updated,
					// we can just give the node the vector of connections
					// and it can update its value.
					if (n.get_layer() == working_layer) n.calculate_value(genome);
				}
			}
		}
		
		// collect the outputs
		for (const Node& n : nodes) {
			if (n.get_node() >= inputs && n.get_node() < inputs + outputs) { output_data[n.get_node() - inputs] = n.get_value(); }
		}

		return output_data;
	}

	void Network::simulate(std::shared_ptr<Simulator> sim, uint32_t steps)
	{
		for (uint32_t i = 0; i < steps; ++i) {
			calculate(sim->get_state());
			sim->update(output_data);
		}
		fitness = sim->get_fitness();
	}

	void Network::mutate_add_node(System& sys)
	{
		// the index of the connection to split
		uint32_t index = uint32_t(System::rand_dist(System::rand_gen) * genome.size());
		if (index == genome.size()) index--;

		// IMPORTANT: 30/04/2022 I believe that it is fine to add nodes in recursive connections
		// including connections of the form (n, n). Testing required once calculate implemented

		bool recursive = genome[index].recursive;

		// if splitting a recursive connection, both the new made connections must be recursive
		Connection in{ genome[index].node1, node_num, true, 1, 0, recursive };
		in.innov_num = sys.get_innov_number(in);

		Connection out{ node_num, genome[index].node2, true, genome[index].weight, 0, recursive };
		out.innov_num = sys.get_innov_number(out);

		// if splitting a recursive connection which is around one node, the input connection should NOT be recursive
		in.recursive = false;

		genome.emplace_back(in);
		genome.emplace_back(out);

		nodes.emplace_back(Node{ node_num, 0, {genome[index].node1} });
		std::find_if(nodes.begin(), nodes.end(), [&](const Node& n) {return n.get_node() == genome[index].node2; })->add_input(node_num);

		genome[index].enabled = false;
		node_num++;

		configure_layers();
	}

	void Network::mutate_add_connection(System& sys, double err)
	{
		// select random nodes to connect
		const Node& n1 = nodes[random_int(nodes.size() - 1)];
		const Node& n2 = nodes[random_int(nodes.size() - 1)];

		if (n2.get_layer() == 0) return; // don't output to a network input
		if (n1.get_layer() == max_layer && n2.get_layer() == max_layer) return; // we're trying to connect two output nodes
		if (std::find(genome.begin(), genome.end(), Connection{ n1.get_node(), n2.get_node() }) != genome.end()) return; // it's already in the genome

		bool recursive = false;
		auto reverse_conn = std::find(genome.begin(), genome.end(), Connection{ n2.get_node(), n1.get_node() });
		if (n1.get_node() == n2.get_node()) recursive = true;
		else if (n1.get_layer() > n2.get_layer()) recursive = true;
		else if (reverse_conn != genome.end()) {
			recursive = !reverse_conn->recursive;
		}
		Connection c{ n1.get_node(), n2.get_node(), true, random(err), 0, recursive };
		c.innov_num = sys.get_innov_number(c);
		genome.push_back(c);

		for (Node& n : nodes) {
			if (n.get_node() == c.node2) n.add_input(c.node1);
		}

		configure_layers();
	}

	void Network::mutate_weights(double mutate_uniform, double err)
	{
		for (Connection& c : genome) {
			if (System::rand_dist(System::rand_gen) <= mutate_uniform) {
				c.weight += random(err);
			}
			else c.weight = random(err);
		}
	}

	Network Network::derive_from_genome(const std::vector<Connection>& genome, uint32_t inputs, uint32_t outputs)
	{
		std::vector<uint32_t> nodes_numbers;
		for (const Connection& c : genome) {
			if (std::find(nodes_numbers.begin(), nodes_numbers.end(), c.node1) == nodes_numbers.end())
				nodes_numbers.push_back(c.node1);

			if (std::find(nodes_numbers.begin(), nodes_numbers.end(), c.node2) == nodes_numbers.end())
				nodes_numbers.push_back(c.node2);
		}

		std::vector<Node> new_nodes;
		for (const uint32_t node_number : nodes_numbers) {
			std::vector<uint32_t> inputs;
			for (const Connection& c : genome) {
				if (c.node2 == node_number) inputs.push_back(c.node1);
			}
			new_nodes.emplace_back(Node{ node_number, 0, inputs });
		}

		std::sort(new_nodes.begin(), new_nodes.end(), [](const Node& a, const Node& b) {return a.get_node() < b.get_node(); });

		Network new_net{ *std::max_element(nodes_numbers.begin(), nodes_numbers.end()) + 1, inputs, outputs};
		new_net.genome = genome;
		new_net.nodes = new_nodes;

		new_net.configure_layers();

		return new_net;
	}

	void Network::mutate(System& s, double node_mut, double conn_mut, double weight_mut, double mut_uniform, double err)
	{
		if (System::rand_dist(System::rand_gen) < node_mut) mutate_add_node(s);
		if (System::rand_dist(System::rand_gen) < conn_mut) mutate_add_connection(s, err);
		if (System::rand_dist(System::rand_gen) < weight_mut) mutate_weights(mut_uniform, err);
	}

	void Network::configure_layers()
	{
		std::vector<uint32_t> set(inputs); // the acceptible set of neurons that neurons in a given layer can have (non-recursive, enabled) connections to
		std::vector<uint32_t> temp_set; // stores the nodes foud to be in a layer before they are added to the main set
		for (uint32_t i = 0; i < inputs; ++i) { set[i] = i; }

		uint32_t layer = 1;
		bool sorting = true; // am I done configuring?

		while (sorting) {
			sorting = false;
			for (Node& n : nodes)
			{
				if (n.get_node() >= inputs + outputs) {
					n.update_back_inputs(genome);
					if (is_subset(set, n.get_back_inputs()) && std::find(set.begin(), set.end(), n.get_node()) == set.end())
					{
						sorting = true;
						n.set_layer(layer);
						temp_set.push_back(n.get_node());
					}
				}
			}

			for (const uint32_t& i : temp_set) {
				if (std::find(set.begin(), set.end(), i) == set.end()) set.push_back(i);
			}

			temp_set.clear();
			layer++;
		}

		// set up the output layer nodes to all have the same layer
		for (Node& n : nodes)
		{
			if (n.get_node() >= inputs && n.get_node() < inputs + outputs) { n.set_layer(layer-1); }
		}

		max_layer = layer - 1;
	}
}
