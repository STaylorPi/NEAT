#include "system.h"
#include "network.h"
#include "xor_test.h"

#include <fstream>
#include <string>
#include <thread>
#include <mutex>
#include <Windows.h>

#define XOR_TEST 0

bool test(const short vk)
{
	if (GetKeyState(vk) & 0x8000) return true;
	return false;
}


void run_NEAT(NEAT::System* sys, double converge, uint32_t sims, std::ostream* os, std::mutex* lock)
{
	lock->lock();
	while (sys->get_max_fitness() < converge)
	{
		uint32_t gens = ((sys->get_generation()/5 + 1) < 5000) ? sys->get_generation() + 1 : 5000;
		sys->simulate_multithread(sims);
		lock->unlock();

		sys->log(*os);

		lock->lock();
		sys->produce_next_generation();
		sys->reset_simulators();
	}
	lock->unlock();
}

#if XOR_TEST == 0
#include "cart_beam.h"
#include "render.h"
#include <SDL.h>

int main(int argc, char* argv[])
{
	try {
		Cart_beam_system test;
	
		NEAT::System sys{ 500, 4, 1, 1 };
		Game render{ "NEAT Cart Beam Testing", 100, 100, 800, 800, false, sys.get_population()[0]};
		NEAT::initialise_system<Cart_beam_system>(sys, test);

		std::mutex lock;
		std::thread thread_test(run_NEAT, &sys, 19000, 5000, &std::cout, &lock);
		
		int frame_delay = 1000 / 60;
		uint32_t frame_start = 0;
		uint32_t frame_end = 0;
		int frame_time = 0;

		while (render.running()) {
			frame_start = SDL_GetTicks();

			render.handle_events();
			render.update(sys, lock);
			render.render();

			frame_end = SDL_GetTicks();

			frame_time = frame_start - frame_end;
			if (frame_time < frame_delay) {
				SDL_Delay(frame_delay - frame_time);
			}
		}

		thread_test.join();

		return 0;
	}
	catch (std::exception& e) {
		std::cout << "Error: " << e.what() << std::endl;
		return 1;
	}
	catch (...) {
		std::cout << "Uncaught error." << std::endl;
		return 2;
	}
}
#else
int main() {
	try {
		XOR test;
		NEAT::System xor_sys{ 500, 3, 1, 1 };
		NEAT::initialise_system<XOR>(xor_sys, test);

		std::mutex lock;
		std::thread xor_exec(run_NEAT, &xor_sys, 3.9999, 4, &std::cout, &lock);
		xor_exec.join();

		return 0;
	}
	catch (std::exception& e) {
		std::cout << "Error: " << e.what() << std::endl;
		return 1;
	}
	catch (...) {
		std::cout << "Uncaught error." << std::endl;
		return 2;
	}
}
#endif