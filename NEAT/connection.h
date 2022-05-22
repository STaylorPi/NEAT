#pragma once

#include <iostream>
#include <iomanip>

namespace NEAT {
	struct Connection {
		Connection(unsigned int node1_, unsigned int node2_)
			:node1{ node1_ }, node2{ node2_ }, enabled{ false }, recursive{ false }, weight{ 0.0 }, value{ 0.0 }, innov_num{ 0 } {}
		Connection(unsigned int node1_, unsigned int node2_, bool enabled_, uint32_t innov_num, bool recursive_)
			:node1{ node1_ }, node2{ node2_ }, enabled{ enabled_ }, weight{ 0.0 }, value{ 0.0 }, innov_num{ innov_num }, recursive{ recursive_ } {}
		Connection(unsigned int node1_, unsigned int node2_, bool enabled_, double weight_, uint32_t innov_num, bool recursive_)
			:node1{ node1_ }, node2{ node2_ }, enabled{ enabled_ }, weight{ weight_ }, value{ 0.0 }, innov_num{ innov_num }, recursive{ recursive_ } {}

		unsigned int node1, node2;
		bool enabled;
		bool recursive;
		double weight;
		double value;

		uint32_t innov_num;

		void update_value(double input) { value = weight * input; }

		bool operator == (const Connection& rhs) const { return (rhs.node1 == node1 && rhs.node2 == node2); } // check if the connection has the same innovation number
		bool operator != (const Connection& rhs) const { return !(rhs.node1 == node1 && rhs.node2 == node2); }

		bool operator > (const Connection& rhs) const { return (rhs.innov_num < innov_num); }
		bool operator >= (const Connection& rhs) const { return (rhs.innov_num <= innov_num); }

		bool operator < (const Connection& rhs) const { return (rhs.innov_num > innov_num); }
		bool operator <= (const Connection& rhs) const { return (rhs.innov_num >= innov_num); }

		friend std::ostream& operator << (std::ostream& os, const Connection& c) {
			os << "n1: " << std::setw(5) << c.node1 << "n2: " << std::setw(5) << c.node2 << "e: " << c.enabled << "w: " << std::setw(12) << c.weight << "I: " << std::setw(12) << c.innov_num << '\n';
			return os;
		}
	};
};