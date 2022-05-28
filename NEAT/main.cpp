#include "system.h"
#include "network.h"
#include "xor_test.h"

int main()
{
	try {
		NEAT::System sys{ 100, 3, 1, 10 };
		XOR test;

		NEAT::initialise_system<XOR>(sys, test);
		__debugbreak();

		return 0;
	}
	catch (std::exception& e) {
		std::cout << "Error: " << e.what() << std::endl;
	}
	catch (...) {
		std::cout << "Uncaught error." << std::endl;
	}
}