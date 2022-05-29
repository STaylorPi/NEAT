#include "system.h"
#include "network.h"
#include "xor_test.h"
#include "cart_beam.h"

#include <fstream>

int main()
{
	try {
		NEAT::System sys{ 150, 5, 1, 0 };
		Cart_beam_system test;

		NEAT::initialise_system<Cart_beam_system>(sys, test);
		std::ofstream outfile{ "out.dat" };

		for (size_t i = 0; i < 1000; i++)
		{
			sys.simulate_population(5000);
			sys.log(outfile);
			sys.produce_next_generation();
			sys.reset_simulators();
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