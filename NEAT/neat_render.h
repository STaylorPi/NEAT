#pragma once
#include "connection.h"
#include "network.h"
#include "system.h"

#include <SDL.h>

namespace NEAT {
	uint32_t nodes_in_layer(const Network&, uint32_t layer);
	uint32_t max_nodes_in_layer(const Network&);
	SDL_Point get_node_pos(SDL_Rect*, const Network&, uint32_t node);

	void render_connection_weight(SDL_Renderer*, SDL_Rect*, const Network&, const Connection&);
	void render_connection_value(SDL_Renderer*, SDL_Rect*, const Network&, const Connection&);
	void render_network(SDL_Renderer*, SDL_Rect*, const Network&);
	void render_fittest(SDL_Renderer*, SDL_Rect*, const System&);
}