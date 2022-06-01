#include "system.h"

namespace NEAT {
	//std::default_random_engine System::rand_gen((std::random_device())());
	std::default_random_engine System::rand_gen;
	std::uniform_real_distribution<double> System::rand_dist(0, 1);
	double modified_sigmoid(double input) { return 1 / (1 + exp(-4.9 * input)); }
	double act_func(double input) { return modified_sigmoid(input); }
	double random(double thresh) { return (System::rand_dist(System::rand_gen) - 0.5) * 2 * thresh; }
	uint32_t random_int(uint32_t ulim) { return uint32_t(System::rand_dist(System::rand_gen) * ulim); }

	System::System(uint32_t size, uint32_t inputs, uint32_t outputs, double err)
		:inputs{ inputs }, outputs{ outputs }, size{ size }, spec_thresh{ 3.0 },
		spec_c1{ 2.0 }, spec_c2{ 2.0 }, spec_c3{ 1.2 }, keep{ .2 },
		node_mut{ 0.03 }, conn_mut{ 0.05 }, weight_mut{ 0.8 }, mut_uniform{ 0.98 }, weight_err{ 1.0 },
		generation{}, crossover_rate{ 0.8 }, disable_thresh{ 0.75 }, target_species{ 20 },
		mean_fitness{}, mean_hidden_nodes{}, max_fitness{}, stagnation_gen{ 10 }
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
				if (!species_found && test_net.speciate(spec_c1, spec_c2, spec_c3, net.get_genome(), spec_thresh)) {
					species_found = true;
					net.set_species(test_net.get_species());
				}
			}
			
			// no species found to match, create a new species with this genome as representative
			if (!species_found) {
				net.set_species(species_reps.size());
				species_reps.push_back(net);
			}
		}

		species_count.clear();
		std::sort(population.begin(), population.end(),
			[](const Network& a, const Network& b) {
				return a.get_species() < b.get_species(); });

		// maybe some species no longer have networks in them
		// so for instance, in the sorted population, species go 4, 4, 4, 6, 6, 7, 7 etc.
		// we want to update the species_count vector acturately, based upon 4, 4, 4, 5, 5, 6, 6 etc. from above
		uint32_t counter = 0; // the number of changes of species index so far
		std::vector<uint32_t> new_species; // size of population, records the correct species of each net
		for (uint32_t i = 0; i < population.size(); ++i) {
			if (i != 0 && (population[i].get_species() != population[i - 1].get_species())) {
				counter++;
			}
			new_species.push_back(counter);
		}

		for (uint32_t i = 0; i < population.size(); ++i) {
			population[i].set_species(new_species[i]);
		}

		// update the species_count vector
		for (uint32_t i = 0; i <= counter; ++i) {
			species_count.push_back(std::count_if(population.begin(), population.end(), [i](const Network& net) {return net.get_species() == i; }));
		}

		spec_thresh -= 0.1 * (int(target_species) - int(species_count.size()));
		spec_thresh = std::clamp(spec_thresh, 0.5, 100.0);
		//if (generation > 10 && species_count.size() == 1) __debugbreak();
	}

	void System::update_reps()
	{
		std::sort(population.begin(), population.end(),
			[](const Network& a, const Network& b) {
				return a.get_species() < b.get_species(); });

		species_reps.clear();
		uint32_t spec_index = 0;
		for (uint32_t i = 0; i < species_count.size(); ++i) {
			species_reps.push_back(population[spec_index + random_int(species_count[i] - 1)]);
			spec_index += species_count[i];
		}
	}

	void System::fitness_sharing()
	{
		for (Network& net : population) {
			net.adjust_fitness(*this);
		}
	}

	void System::update_fitness_log()
	{
		uint32_t spec_index = 0;
		for (uint32_t spec = 0; spec < species_count.size(); ++spec) {
			if (fitness_trends.size() == spec) {
				fitness_trends.push_back(std::vector<double>());
			}
			double spec_average = 0;
			for (uint32_t neti = 0; neti < species_count[spec]; ++neti) {
				spec_average += population[spec_index + neti].get_raw_fitness() / species_count[spec];
			}
			fitness_trends[spec].push_back(spec_average);
			spec_index += species_count[spec];
		}

		while (fitness_trends.size() > species_count.size()) fitness_trends.pop_back();
		//if (species_count.size() > 1 && generation > 10) __debugbreak();
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
			//if (species_offspring[species_offspring.size() - 1] == 0) __debugbreak();

			/*// if the species hasn't improved in 15 generations
			if (fitness_trends[spec].size() >= stagnation_gen && generation % 15 == 0) {
				if (fitness_trends[spec][fitness_trends[spec].size() - 1] -
					fitness_trends[spec][fitness_trends[spec].size() - stagnation_gen] < 0) {
					species_offspring[spec] = 0;
					fitness_trends.erase(fitness_trends.begin() + spec);
					std::cout << "Culled species " << spec << '\n';
				}
			}*/
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

	void System::simulate_subset(System* s, uint32_t first, uint32_t last, uint32_t steps)
	{
		for (uint32_t i = first; i < last; ++i) {
			s->population[i].simulate(s->simulators[i], steps);
		}
	}

	void System::produce_next_generation()
	{
		speciate();
		update_reps();
		fitness_sharing();
		//update_fitness_log();
		std::vector<uint32_t> species_offspring = assign_offspring();
		cull_population();

		// now in a state to produce the next generation
		std::vector<Network> new_population;
		std::vector<Network> copy_unchanged; // the best nets from the species with 5 or more

		uint32_t spec_index = 0;
		for (uint32_t spec = 0; spec < species_offspring.size(); ++spec) {
			uint32_t spec_len = species_count[spec] - uint32_t(species_count[spec] * (1 - keep)); // before amount - amount culled
			if (species_count[spec] >= 5) {
				copy_unchanged.push_back(population[spec_index + spec_len - 1]);
				species_offspring[spec]--;
			}

			for (uint32_t i = 0; i < species_offspring[spec]; ++i) {
				if (System::rand_dist(System::rand_gen) > crossover_rate) // mutation without crossover: copy random one
					new_population.push_back(population[spec_index + random_int(spec_len - 1)]);

				else {
					uint32_t lnet = spec_index + random_int(spec_len - 1);
					uint32_t rnet = spec_index + random_int(spec_len - 1);
					new_population.push_back(population[lnet].cross(population[rnet], disable_thresh));
				}
			}
			spec_index += spec_len;
		}
		
		for (Network& net : new_population) {
			net.mutate(*this, node_mut, conn_mut, weight_mut, mut_uniform, weight_err);
		}

		for (const Network& net : copy_unchanged) {
			new_population.push_back(net);
		}

		// if there are species that were not allowed to reproduce
		while (new_population.size() < size) {
			new_population.push_back(Network{ *this, inputs, outputs, weight_err });
		}

		population = new_population;
		generation++;
	}

	std::ostream& System::log(std::ostream& os)
	{
		mean_fitness = 0;
		mean_hidden_nodes = 0;
		for (const Network& net : population) {
			mean_fitness += net.get_raw_fitness() / size;
			mean_hidden_nodes += double(net.get_hidden_nodes()) / size;
		}

		max_fitness = std::max_element(population.begin(), population.end(), [](const Network& a, const Network& b)
			{return a.get_raw_fitness() < b.get_raw_fitness(); })->get_raw_fitness();

		os << "====GENERATION " << generation << "====\n";
		os << "Mean fitness:      " << mean_fitness << '\n';
		os << "Mean hidden nodes: " << mean_hidden_nodes << '\n';
		os << "Species:           " << species_count.size() << '\n';
		os << "Spec. Threshold:   " << spec_thresh << '\n';
		os << "Max fitness:       " << max_fitness << "\n";
		os << "Genes:             " << genes.size() << "\n\n";

		if (&os != &std::cout) {
			std::cout << generation << '\r';
		}

		return os;
	}

	std::ostream& System::dump_fittest(std::ostream& os)
	{
		Network& fittest = *std::max_element(population.begin(), population.end(), [](const Network& a, const Network& b)
			{return a.get_raw_fitness() < b.get_raw_fitness(); });
		return fittest.byte_genome_dump(os);
	}

	void System::simulate_population(uint32_t timesteps)
	{
		for (uint32_t i = 0; i < size; ++i) {
			population[i].simulate(simulators[i], timesteps);
		}
	}

	void System::simulate_multithread(uint32_t timesteps)
	{
		uint32_t cores = std::thread::hardware_concurrency();
		if (cores == 0) cores = 8;
		const uint32_t num = size / cores; // # of population to run on each core

		std::vector<std::thread> threads;
		for (uint32_t i = 0; i < cores; ++i) {
			uint32_t first = i * num;
			uint32_t last = (i + 1) * num;
			if ((i + 1) == cores) last = size;
			threads.emplace_back(std::thread(&System::simulate_subset, this, first,
				last, timesteps));
		}

		for (std::thread& t : threads) t.join();
	}


	void System::reset_simulators()
	{
		for (auto& sim : simulators) {
			sim->reset();
		}
	}
}