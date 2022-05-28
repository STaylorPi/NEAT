#include "system.h"

namespace NEAT {
	std::default_random_engine System::rand_gen;
	std::uniform_real_distribution<double> System::rand_dist(0, 1);
	double modified_sigmoid(double input) { return 1 / (1 + exp(-4.9 * input)); }
	double act_func(double input) { return modified_sigmoid(input); }
	double random(double thresh) { return (System::rand_dist(System::rand_gen) - 0.5) * 2 * thresh; }
	uint32_t random_int(uint32_t ulim) { return uint32_t(System::rand_dist(System::rand_gen) * ulim); }

	System::System(uint32_t size, uint32_t inputs, uint32_t outputs, double err)
		:inputs{ inputs }, outputs{ outputs }, size{ size }, spec_thresh{ .1 }, 
		spec_c1{ 1.0 }, spec_c2{ 1.0 }, spec_c3{ .4 }, keep{ .5 }
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

	void System::update_reps()
	{
		std::sort(population.begin(), population.end(),
			[](const Network& a, const Network& b) {
				return a.get_species() < b.get_species(); });

		uint32_t spec_index = 0;
		for (uint32_t i = 0; i < species_count.size(); ++i) {
			species_reps[i] = population[spec_index + random_int(species_count[i])];
			spec_index += species_count[i];
		}
	}

	void System::fitness_sharing()
	{
		for (Network& net : population) {
			net.adjust_fitness(*this);
		}
	}

	std::vector<uint32_t> System::assign_offspring()
	{
		std::vector<uint32_t> species_offspring;
		double average_fitness = 0;
		for (Network& net : population) {
			average_fitness += net.get_shared_fitness() / size;
		}

		for (uint32_t spec = 0; spec < species_count.size(); ++spec) {
			double spec_fitness = 0;
			for (const Network& net : population) {
				if (net.get_species() == spec) spec_fitness += net.get_shared_fitness() / species_count[spec];
			}
			// compute the number of offspring for the species
			species_offspring.push_back(uint32_t(round(spec_fitness / average_fitness * species_count[spec])));
		}

		// make sure the population size is always constant
		int diff = size - std::accumulate(species_offspring.begin(), species_offspring.end(), 0);
		
		if (diff > 0) {
			for (; diff > 0; --diff) {
				species_offspring[random_int(species_offspring.size() - 1)]++;
			}
		}
		else if (diff < 0) {
			for (; diff < 0; ++diff) {
				species_offspring[random_int(species_offspring.size() - 1)]--;
			}
		}

		return species_offspring;
	}

	void System::cull_population()
	{
		std::vector<uint32_t> to_cull; // the number to keep from each species
		for (uint32_t count : species_count) {
			to_cull.push_back(uint32_t(count * (1 - keep)));
		}

		std::sort(population.begin(), population.end(),
			[](const Network& a, const Network& b) {
				if (a.get_species() < b.get_species()) return true;
				else if (a.get_species() > b.get_species()) return false;
				else return a.get_shared_fitness() < b.get_shared_fitness();
			});

		std::vector<Network> culled_population;
		uint32_t spec_index = 0;
		for (uint32_t spec = 0; spec < species_count.size(); ++spec) {
			uint32_t lower_bound = spec_index + to_cull[spec];

			// number from this species to keep
			uint32_t num = species_count[spec] - to_cull[spec];
			for (uint32_t i = 0; i < num; ++i) {
				culled_population.push_back(population[i + lower_bound]);
			}
			spec_index += species_count[spec];
		}

		population = culled_population;
	}

	void System::simulate_population(uint32_t timesteps)
	{
		for (uint32_t i = 0; i < size; ++i) {
			population[i].simulate(simulators[i], timesteps);
		}
	}
}