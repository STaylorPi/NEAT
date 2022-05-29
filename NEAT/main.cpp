#include "system.h"
#include "network.h"
#include "xor_test.h"

#include <fstream>

int main()
{
	try {
		NEAT::System sys{ 150, 3, 1, 0 };
		XOR test;

		NEAT::initialise_system<XOR>(sys, test);
		std::ofstream outfile{ "out.dat" };

		for (size_t i = 0; i < 300; i++)
		{
			sys.simulate_population(4);
			sys.produce_next_generation();
			sys.reset_simulators();
			sys.log(outfile);
		}

		return 0;
	}
	catch (std::exception& e) {
		std::cout << "Error: " << e.what() << std::endl;
	}
	catch (...) {
		std::cout << "Uncaught error." << std::endl;
	}
}