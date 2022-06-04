#pragma once
#include <cmath>
#include <iostream>
#include <iomanip>
#include <SDL.h>
#include <math.h>

#include "simulator.h"

class Cart_beam_system : public NEAT::Simulator {
public:
	Cart_beam_system()
		:x{ 0 }, x_dot{ 0 }, x_double_dot{ 0 }, old_x{ 0 }, old_x_dot{ 0 }, phi{ M_PI }, phi_dot{0},
		phi_double_dot{ 0 }, mass_cart{ 1 }, mass_rod{ 1 }, length_rod{ 1 }, drag{ 0.1 }, rail_limit{ 3 }, fitness{},
		x_desired{ 1 }, timesteps{}, generation{} {}
	Cart_beam_system(double mr, double mc, double length, double drag_, double rail_limit_)
		:x{ 0 }, x_dot{ 0 }, x_double_dot{ 0 }, old_x{ 0 }, old_x_dot{ 0 }, phi{ 0 }, phi_dot{ 0 }, phi_double_dot{ 0 },
		mass_cart{ mc }, mass_rod{ mr }, length_rod{ length }, drag{ drag_ }, rail_limit{ rail_limit_ }, fitness{}, x_desired{1},
		timesteps{}, generation{} {}

	void calculate_values(double ts);
	void calculate_values(double ts, double x_ddot);

	void SDL_draw(SDL_Renderer* ren);

	void update_with_network_output(const std::vector<double>& net_out) override {
		calculate_values(0.01, 5*(net_out[0]-0.5));
		update_fitness();
		//x_desired = 0.0;
		x_desired = (generation / 1000.0) * sin(0.01 * timesteps);
		timesteps++;
	}

	const std::vector<double>& get_inputs_to_network() override {
		inputs.clear();

		double in_phi = fmod(phi-M_PI, 2 * M_PI)/M_PI;

		inputs.push_back(in_phi);
		inputs.push_back(phi_dot);
		inputs.push_back(x - x_desired);
		return inputs;
	}

	void reset() override {
		x = 0;
		x_dot = 0;
		x_double_dot = 0;
		old_x = 0;
		old_x_dot = 0;
		phi = M_PI;
		phi_dot = 0;
		phi_double_dot = 0;
		fitness = 0;
		x_desired = 0;
		timesteps = 0;
		generation++;
	}
	
	double get_fitness() override { return fitness; }

	friend std::ostream& operator<<(std::ostream& os, const Cart_beam_system& cbs)
	{
		os << "phi: " << std::setw(15) << cbs.phi << '\n';
		return os;
	}

	double get_x() const { return x; }

private:
	double x, x_dot, x_double_dot, phi, phi_dot, phi_double_dot;
	double old_x, old_x_dot;
	double mass_cart, mass_rod;
	double length_rod;
	double drag;
	double rail_limit;
	double x_desired;
	uint64_t timesteps;
	uint32_t generation;

	double fitness;
	std::vector<double> inputs;

	void set_x_dot(double xdt) { old_x_dot = x_dot;  x_dot = xdt; }
	void set_x_ddot(double xddt) { x_double_dot = xddt; }
	void update_fitness() {
		fitness += 1 / ((5 * sqrt(pow(length_rod * sin(phi) + (x - x_desired), 2) + pow(length_rod * (1 + cos(phi)), 2)) + 1) * (1 + abs(phi_dot / 10)));
	}
};