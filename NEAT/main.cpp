#include "network.h"

void not_safe() { int* a = new int; }

int main()
{
	try {
		NEAT::System sys{ 1, 5, 2 };
		sys.init();
		for (int i=0; i<400; ++i)
			sys.get_population()[0].mutate(sys, 0.03, 0.05, 0.8, 0.9, 0.5);

		NEAT::Network test = NEAT::Network::derive_from_genome(sys.get_population()[0].get_genome(), 5, 2);
		//__debugbreak();

		return 0;
	}
	catch (std::exception& e) {
		std::cout << "Error: " << e.what() << std::endl;
	}
	catch (...) {
		std::cout << "Uncaught error." << std::endl;
	}
}