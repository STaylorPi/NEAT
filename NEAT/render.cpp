#pragma once
#include "render.h"

SDL_Renderer* Game::renderer = nullptr;

Game::Game(const char* title, int xpos, int ypos, int width_, int height_, bool fullscreen, const NEAT::Network& n)
    :width{ width_ }, height{ height_ }, cbs{}, net{ n }
{
    init(title, xpos, ypos, width, height, fullscreen);
}

Game::~Game()
{
    clean();
}

void Game::init(const char* title, int xpos, int ypos, int width, int height, bool fullscreen)
{
    if (SDL_Init(SDL_INIT_EVERYTHING) == 0)
    {
        std::cout << "subsystem init succcess!" << std::endl;

        int flags = 0;
        if (fullscreen) flags = SDL_WINDOW_FULLSCREEN;
        window = SDL_CreateWindow(title, xpos, ypos, width, height, flags);
        if (window)
        {
            std::cout << "window created!" << std::endl;
        }

        renderer = SDL_CreateRenderer(window, -1, 0);
        if (renderer)
        {
            std::cout << "renderer created!" << std::endl;
        }

        is_running = true;
    }
    else {
        is_running = false;
        throw std::runtime_error("Initialisation error");
    }
}

void Game::handle_events()
{
    SDL_Event event;
    SDL_PollEvent(&event);
    switch (event.type) {
    case SDL_QUIT:
        is_running = false;
        break;

    case SDL_MOUSEMOTION:
        int x, y;
        SDL_GetMouseState(&x, &y);
        x -= 400;
        break;

    case SDL_KEYDOWN:
        switch (event.key.keysym.sym) {
        case SDLK_n:
            requesting_net = true;
        }

    default:
        break;
    }
}

void Game::update(NEAT::System& sys, std::mutex& lock)
{
    if (requesting_net) {
        lock.lock();
        net = *std::max_element(sys.get_population().begin(), sys.get_population().end(),
            [](const NEAT::Network& a, const NEAT::Network& b) {return a.get_raw_fitness() < b.get_raw_fitness(); });
        lock.unlock();
        net.configure_layers();
        requesting_net = false;
        cbs.reset();
    }

    net.calculate(cbs.get_inputs_to_network());
    cbs.update_with_network_output(net.get_output());
}

void Game::render()
{
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);
    cbs.SDL_draw(renderer);

    SDL_Rect net_rect;
    net_rect.x = 100;
    net_rect.y = 100;
    net_rect.w = 600;
    net_rect.h = 150;
    NEAT::render_network(renderer, &net_rect, net);
    SDL_RenderPresent(renderer);
}

void Game::clean()
{
    SDL_DestroyWindow(window);
    SDL_DestroyRenderer(renderer);
    SDL_Quit();
    std::cout << "game cleaned" << std::endl;
}

void Game::mainloop(uint32_t fps)
{
}
