#include "neat_render.h"

uint32_t NEAT::nodes_in_layer(const Network& net, uint32_t layer)
{
    return std::count_if(net.get_nodes().begin(),
        net.get_nodes().end(),
        [layer](const Network::Node& n) { return n.get_layer() == layer; });
}

uint32_t NEAT::max_nodes_in_layer(const Network& net)
{
    uint32_t max_nodes = 0;
    for (uint32_t layer = 0; layer < net.get_max_layer(); ++layer) {
        if (max_nodes < nodes_in_layer(net, layer))
            max_nodes = nodes_in_layer(net, layer);
    }

    return max_nodes;
}

SDL_Point NEAT::get_node_pos(SDL_Rect* rect, const Network& net, uint32_t node)
{
    SDL_Point point;
    auto nodeit = std::find_if(net.get_nodes().begin(),
        net.get_nodes().end(),
        [node](const Network::Node& n) { return n.get_node() == node; });

    if (nodeit == net.get_nodes().end())
        throw std::runtime_error("Invalid node passed to NEAT::get_node_pos");

    Network::Node net_node = *nodeit;

    std::vector<Network::Node> net_nodes = net.get_nodes();
    std::sort(net_nodes.begin(), net_nodes.end(),
        [](const Network::Node& a,
            const Network::Node& b) { return a.get_node() < b.get_node(); });

    uint32_t first_node_in_layer = std::find_if(net_nodes.begin(), net_nodes.end(),
        [net_node](const Network::Node& n)
        {return n.get_layer() == net_node.get_layer(); })->get_node();

    uint32_t layer_num = nodes_in_layer(net, net_node.get_layer());
    uint32_t layer_space = rect->h / max_nodes_in_layer(net);

    point.x = rect->x + int(rect->w * (double(net_node.get_layer()) / net.get_max_layer()));
    point.y = rect->h / 2 + (net_node.get_node() - first_node_in_layer - (layer_num + 1) / 2) *
        layer_space + rect->y;

    return point;
}

void NEAT::render_connection_weight(SDL_Renderer* ren, SDL_Rect* rect, const Network& net, const Connection& c)
{
    SDL_Point p1 = get_node_pos(rect, net, c.node1);
    SDL_Point p2 = get_node_pos(rect, net, c.node2);

    // colour
    int r = (c.weight > 0) * 100 * c.weight;
    int b = - (c.weight < 0) * 100 * c.weight;
    r = std::clamp(r, 0, 255);
    b = std::clamp(b, 0, 255);
    if (c.recursive) SDL_SetRenderDrawColor(ren, r, 100, b, 255);
    else SDL_SetRenderDrawColor(ren, r, 0, b, 255);
    SDL_RenderDrawLine(ren, p1.x, p1.y, p2.x, p2.y);
}

void NEAT::render_connection_value(SDL_Renderer* ren, SDL_Rect* rect, const Network& net, const Connection& c)
{
    SDL_Point p1 = get_node_pos(rect, net, c.node1);
    SDL_Point p2 = get_node_pos(rect, net, c.node2);

    // colour
    int r = (c.value > 0) * 100 * c.value;
    int b = -(c.value < 0) * 100 * c.value;
    r = std::clamp(r, 0, 255);
    b = std::clamp(b, 0, 255);
    if (c.recursive) SDL_SetRenderDrawColor(ren, r, 100, b, 255);
    else SDL_SetRenderDrawColor(ren, r, 0, b, 255);
    SDL_RenderDrawLine(ren, p1.x, p1.y, p2.x, p2.y);
}

void NEAT::render_network(SDL_Renderer* ren, SDL_Rect* rect, const Network& net)
{
    SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
    SDL_Point p;
    SDL_Rect r;
    //if (net.get_max_layer() > 1) __debugbreak();
    for (const auto& n : net.get_nodes()) {
        p = get_node_pos(rect, net, n.get_node());
        r.x = p.x - 5;
        r.y = p.y - 5;
        r.w = 10;
        r.h = 10;
        SDL_RenderFillRect(ren, &r);
    }

    for (const auto& c : net.get_genome()) {
        render_connection_value(ren, rect, net, c);
    }
}
