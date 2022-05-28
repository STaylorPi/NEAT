#include "system.h"

namespace NEAT {
	std::default_random_engine System::rand_gen;
	std::uniform_real_distribution<double> System::rand_dist(0, 1);
	double modified_sigmoid(double input) { return 1 / (1 + exp(-4.9 * input)); }
	double act_func(double input) { return modified_sigmoid(input); }
	double random(double thresh) { return (System::rand_dist(System::rand_gen) - 0.5) * 2 * thresh; }
	uint32_t random_int(uint32_t ulim) { return uint32_t(System::rand_dist(System::rand_gen) * ulim); }

	System::System(uint32_t size, uint32_t inputs, uint32_t outputs, double err)
		:inputs{ inputs }, outputs{ outputs }, size{ size }, spec_thresh{ 3.0 }, 
		spec_c1{ 1.0 }, spec_c2{ 1.0 }, spec_c3{ 0.4 }
	{
		for (uint32_t i = 0; i < size; ++i) {
			population.emplace_back(Network{ *this, inputs, outputs, err });
		}
	}

	void System::init_simulators(const std::vector<std::shared_ptr<Simulator>>& sims)
	{
		if (sims.size() != size) throw std::runtime_error("Incorrect simulator length");
		simulators = sims;
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
			return genes.size() - 1;
		}
	}

	void System::speciate()
	{
		for (Network& net : population) {
			bool species_found = false;
			for (const Network& test_net : species_reps) {
				if (test_net.speciate(spec_c1, spec_c2, spec_c3, net.get_genome(), spec_thresh)) {
					species_found = true;
					net.set_species(test_net.get_species());
					species_count[test_net.get_species()]++;
					break;
				}
			}
			
			// no species found to match, create a new species with this genome as representative
			if (!species_found) {
				net.set_species(species_reps.size());
				species_reps.push_back(net);
				species_count.push_back(1);
			}
		}
	}

	void System::fitness_sharing()
	{
		for (Network& net : population) {
			net.adjust_fitness(*this);
		}
	}

	void System::simulate_population(uint32_t timesteps)
	{
		for (uint32_t i = 0; i < size; ++i) {
			population[i].simulate(simulators[i], timesteps);
		}
	}

}