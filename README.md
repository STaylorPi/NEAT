# NEAT

My own implementation of the NEAT algorithm, http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf, originally published in 2002.

The project is still a work in progress, but once finished it will have the following interface:
 - Users will inherit from the interface class in simulator.h, providing overrides for all of the pure virtual functions;
 - A `std::vector<std::shared_ptr<Simulator>>` will be held in each running NEAT system that will pass the pointer to networks for simulation, as well as allowing the user external updates (eg. graphical rendering) without interfering with the underlying NEAT system;
 - The `NEAT::System` class will be the outward interface, and there can be many instances of a NEAT system at a given time.
