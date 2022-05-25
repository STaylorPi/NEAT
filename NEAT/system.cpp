#include "system.h"

namespace NEAT {
	std::default_random_engine System::rand_gen;
	std::uniform_real_distribution<double> System::rand_dist(0, 1);
	double modified_sigmoid(double input) { return 1 / (1 + exp(-4.9 * input)); }
	double act_func(double input) { return modified_sigmoid(input); }
	double random(double thresh) { return (System::rand_dist(System::rand_gen) - 0.5) * 2 * thresh; }
	uint32_t random_int(uint32_t ulim) { return uint32_t(System::rand_dist(System::rand_gen) * ulim); }

	System::System(uint32_t size, uint32_t inputs, uint32_t outputs)
		:inputs{ inputs }, outputs{ outputs }, size{ size }
	{
	}

	void System::init()
	{
		for (uint32_t i = 0; i < size; ++i) {
			population.emplace_back(Network{*this, inputs, outputs});
		}
	}

	uint32_t System::get_innov_number(const Connection& gene)
	{
		auto c = std::find(genes.begin(), genes.end(), gene);
		if (c != genes.end())
		{
			return c->innov_num;
		}

		else {
			genes.emplace_back(Connection{ gene.node1, gene.node2, true, genes.size(), false });
			return genes.size();
		}
	}

}