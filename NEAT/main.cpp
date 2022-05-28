#include "system.h"
#include "network.h"
#include "xor_test.h"

int main()
{
	try {
		NEAT::System sys{ 100, 3, 1, 1 };
		XOR test;

		NEAT::initialise_system<XOR>(sys, test);
		
		sys.simulate_population(4);
		sys.speciate();
		sys.fitness_sharing();
		auto test2 = sys.assign_offspring();
		__debugbreak();
		sys.update_reps();
		__debugbreak();
		sys.cull_population();

		return 0;
	}
	catch (std::exception& e) {
		std::cout << "Error: " << e.what() << std::endl;
	}
	catch (...) {
		std::cout << "Uncaught error." << std::endl;
	}
}