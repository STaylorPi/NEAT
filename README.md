# NEAT

My own implementation of the NEAT algorithm, http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf, originally published in 2002.

The project is still a work in progress, but once finished it will have the following interface:
 - Users will inherit from the interface class in simulator.h, providing overrides for all of the pure virtual functions;
 - A `std::vector<std::shared_ptr<Simulator>>` will be held in each running NEAT system that will pass the pointer to networks for simulation, as well as allowing the user external updates (eg. graphical rendering) without interfering with the underlying NEAT system;
 - The `NEAT::System` class will be the outward interface, and there can be many instances of a NEAT system at a given time.

## To-Do List
 - Add a makefile and macro-ify debug code, as well as adding some profiling code for chrome tracing
 - NEAT::Network::cross implementation
 - NEAT::System initialisation with a template function that can take in any simulator
 - NEAT::System user interface for learning
 - system-wide speciation algorithm (NEAT::Network::speciate already implemented)
 - Explicit fitness sharing
 - Assigning and generating offspring from a generation (best members of species are passed straight through)
