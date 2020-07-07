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
                k = (x, y, o) #x,y pos as well as valid in-orientations
                v = nodes[k]
                for t in Direction:
                    if env.rail.get_transition(k, t):
                        newx, newy = modify_coordinates(x, y, t)
                        v_dest = nodes[(newx, newy, t)]
                        g.add_edge(v, v_dest, length=1)
    for k, v in nodes.items():
        v['x'], v['y'], v['o'] = k
        v['xs'], v['ys'], v['os'] = ([kk] for kk in k)
    empty_vs = [v.index for v in g.vs if not v.all_edges()]
    g.delete_vertices(empty_vs)
    return g

def merge_linear_paths(g):
    vertices_to_delete = []
    for v in g.vs:
        if v.indegree() == 1 and v.outdegree() == 1:
            e1, e2 = v.in_edges()[0], v.out_edges()[0]
            s, t = e1.source_vertex, e2.target_vertex
            g.add_edge(s, t, length=e1['length'] + e2['length'])
            for a in ('xs', 'ys', 'os'):
                s[a].extend(v[a])
                t[a].extend(v[a])
            g.delete_edges([e1, e2])
            vertices_to_delete.append(v)
    g.delete_vertices(vertices_to_delete)


class TransitionGraph:
    def __init__(self, env):
        self.g = build_transition_graph(env)
        merge_linear_paths(g)


def get_linear_path(g,v): #only explores one direction-> start right after junction
    visited=[v]
    while True:
        if len(v.successors())!=1 or is_dead_end(v.successors()[0]) or is_junction(g,v.successors()[0]) or v.successors()[0] in visited: #switch or dead-end or junction or loop round
            break
        v=v.successors()[0]
        visited.append(v)
    return visited


def join_linear_path(g,path):
    assert len(path)>2, "only paths longer than two can be joined"
    g.add_edge(path[0],path[-1],length=len(path)-1)
    return [v.index for v in path[1:-1]] 
   
def is_junction(g,v):    
    x_matches=[i for i, e in enumerate(g.vs['x']) if e == v['x']]
    matches=list(g.vs[x_matches]['y']).count(v['y'])
    return matches>2

def is_dead_end(v):
    return len(v.successors())==1 and abs(v['o'] - v.successors()[0]['o'])==2

