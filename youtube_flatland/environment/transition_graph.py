#!/usr/bin/env python3

import collections

import igraph

from flatland.core.grid.grid4 import Grid4TransitionsEnum as Direction


def modify_coordinates(x, y, t):
    if t == Direction.NORTH:
        return x-1, y
    if t == Direction.SOUTH:
        return x+1, y
    if t == Direction.EAST:
        return x, y+1
    if t == Direction.WEST:
        return x, y-1
    raise ValueError('Unknown direction: {}'.format(t))


def build_transition_graph(env):
    g = igraph.Graph(directed=True)
    nodes = collections.defaultdict(lambda: g.add_vertex())
    for x in range(env.height):
        for y in range(env.width):
            for o in Direction:
                k = (x, y, o)
                v = nodes[k]
                for t in Direction:
                    if env.rail.get_transition(k, t):
                        newx, newy = modify_coordinates(x, y, t)
                        v_dest = nodes[(newx, newy, t)]
                        g.add_edge(v, v_dest)
    for k, v in nodes.items():
        v['x'], v['y'], v['o'] = k
    empty_vs = [v.index for v in g.vs if not v.all_edges()]
    g.delete_vertices(empty_vs)
    return g
