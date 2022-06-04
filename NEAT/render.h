#pragma once

#include <SDL.h>
#include <iostream>
#include <exception>
#include <vector>
#include <mutex>
#include "cart_beam.h"
#include "system.h"
#include "neat_render.h"

class Game {
public:
    Game(const char* title, int xpos, int ypos, int width, int height, bool fullscreen, const NEAT::Network& n);
    ~Game();

    void init(const char* title, int xpos, int ypos, int width, int height, bool fullscreen);

    void handle_events();
    void update(NEAT::System& sys, std::mutex&);
    void render();
    void clean();

    void mainloop(uint32_t fps);

    bool running() { return is_running; }

    static SDL_Renderer* renderer;

private:
    bool is_running;
    SDL_Window* window;

    int count;
    int width, height;

    Cart_beam_system cbs;
    NEAT::Network net;
    bool requesting_net;
};
