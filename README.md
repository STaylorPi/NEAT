# NEAT

My own implementation of the NEAT algorithm, http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf, originally published in 2002.

The project is still a work in progress, but once finished it will have the following interface:
 - Users will inherit from the interface class in simulator.h, providing overrides for all of the pure virtual functions;
 - A `std::vector<std::shared_ptr<Simulator>>` will be held in each running NEAT system that will pass the pointer to networks for simulation, as well as allowing the user external updates (eg. graphical rendering) without interfering with the underlying NEAT system;
 - The `NEAT::System` class will be the outward interface, and there can be many instances of a NEAT system at a given time.

## To-Do List
 1. Add a makefile and macro-ify debug code, as well as adding some profiling code for chrome tracing
 2. NEAT::Network::cross implementation
 3. NEAT::System initialisation with a template function that can take in any simulator
 4. NEAT::System user interface for learning
 5. system-wide speciation algorithm (NEAT::Network::speciate already implemented)
 6. Explicit fitness sharing
 7. Assigning and generating offspring from a generation (best members of species are passed straight through)
