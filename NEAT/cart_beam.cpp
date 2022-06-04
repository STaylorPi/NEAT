#include "cart_beam.h"

void Cart_beam_system::calculate_values(double ts)
{
	if ((x >= rail_limit && x_dot > 0) || (x <= -rail_limit && x_dot < 0)) x_dot = 0;

	x_double_dot = (x_dot - old_x_dot) / ts;
	//x_dot += x_double_dot * ts;
	x += x_dot * ts;

	old_x_dot = x_dot;

	double tension = pow(phi_dot, 2) * length_rod / 2 * mass_rod;
	double cart_force = -x_double_dot * (mass_rod + mass_cart);

	double force_x = cart_force + tension * sin(phi);
	double force_y = -tension * cos(phi) - mass_rod * 9.81;

	double inertia = (1.0 / 3) * mass_rod * pow(length_rod, 2);

	phi_double_dot = (force_x * cos(phi) + force_y * sin(phi) - drag / mass_rod * phi_dot) / inertia;
	phi_dot += phi_double_dot * ts;
	phi += phi_dot * ts;
}

void Cart_beam_system::calculate_values(double ts, double x_ddot)
{
	//if ((x >= rail_limit && x_dot > 0) || (x <= -rail_limit && x_dot < 0)) x_dot = 0;

	x_double_dot = x_ddot;
	x_dot += x_double_dot * ts;
	x += x_dot * ts;

	double tension = pow(phi_dot, 2) * length_rod / 2 * mass_rod;
	double cart_force = -x_double_dot * (mass_rod + mass_cart);

	double force_x = cart_force + tension * sin(phi);
	double force_y = -tension * cos(phi) - mass_rod * 9.81;

	double inertia = (1.0 / 3) * mass_rod * pow(length_rod, 2);

	phi_double_dot = (force_x * cos(phi) + force_y * sin(phi) - drag / mass_rod * phi_dot) / inertia;
	phi_dot += phi_double_dot * ts;
	phi += phi_dot * ts;
}


void Cart_beam_system::SDL_draw(SDL_Renderer* ren)
{
	double xm = length_rod * sin(phi);
	double ym = length_rod * cos(phi);
	xm *= 100;
	ym *= 100;
	xm += 400;
	ym += 600;
	SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
	SDL_RenderDrawLine(ren, int(400 + 100 * x), 600, int(xm + 100 * x), int(ym));
	SDL_SetRenderDrawColor(ren, 200, 200, 200, 255);
	SDL_RenderDrawLine(ren, 400 + 100 * x_desired, 600, 400 + 100 * x_desired, 500);
}
