#include "system.h"
#include "network.h"
#include "xor_test.h"

int main()
{
	try {
		NEAT::System sys{ 150, 3, 1, 0 };
		XOR test;

		NEAT::initialise_system<XOR>(sys, test);

		for (size_t i = 0; i < 1000; i++)
		{
			sys.simulate_population(4);
			sys.produce_next_generation();
			sys.reset_simulators();
			//if (i == 500) __debugbreak();
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