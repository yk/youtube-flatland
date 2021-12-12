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

def create_track_node(g, xyo):
    v = g.add_vertex()
    v['x'], v['y'], v['o'] = xyo
    xy = xyo[:-1]
    v['length'] = 1
    v['xyos'] = [xyo]
    v['xys'] = [xy]
    v['node_type'] = 'track'
    v['target_of'] = set()
    v['resource'] = -1
    return v

def build_transition_graph(env):
    g = igraph.Graph(directed=True)
    nodes = dict()
    def _node(k):
        if k not in nodes:
            nodes[k] = create_track_node(g, k)
        return nodes[k]
    for x in range(env.height):
        for y in range(env.width):
            for o in Direction:
                k = (x, y, o) #x,y pos as well as valid in-orientations
                v = _node( k )
                for t in Direction:
                    if env.rail.get_transition(k, t):
                        newx, newy = modify_coordinates(x, y, t)
                        v_dest = _node( (newx, newy, t) )
                        g.add_edge(v, v_dest, length=1, edge_type='track')

    empty_vs = [v.index for v in g.vs if not v.all_edges()]
    g.delete_vertices(empty_vs)

    ## annotate nodes with extra informations
    for v in g.vs:
        #v['tar_of_agent']=-1 # no target
        #v['pos_of_agent']=-1 #no agent      needed???
        if is_junction(g,v):
            v["node_type"]='junction' 

    return g

def merge_within_junction(g):
    for v in g.vs:
        if v['node_type']=='junction' and v.outdegree()==1:
            x_matches=[i for i, e in enumerate(g.vs['x']) if e == v['x']]
            matches=[x_match for i, x_match in enumerate(x_matches) if g.vs[x_match]['y']==v['y']]
            matches.remove(v.index)
            for i in matches:
                match=g.vs[i]
                if match.outdegree()==1 and v.out_edges()[0].target_vertex==match.out_edges()[0].target_vertex:
                        for e in match.in_edges():
                            g.add_edge(e.source_vertex,v,length=e['length'], edge_type='track')
                            g.delete_vertices(match)

def node_is_mergeable(n):
    if n['node_type'] == 'junction':
        return False
    return True

def merge_linear_paths(g):
    vertices_to_delete = []
    for v in g.vs:
        if v.indegree() == 1 and v.outdegree() == 1 and v['node_type'] == 'track': # by retaining the junctions, we separate different resources (track segments)
            if not node_is_mergeable(v):
                continue
            e1, e2 = v.in_edges()[0], v.out_edges()[0]
            s, t = e1.source_vertex, e2.target_vertex

            if not (node_is_mergeable(s) or node_is_mergeable(t)):
                # we have to retain linear pieces between junctions/targets, because they are separate resources
                continue

            g.add_edge(s, t, length=e1['length'] + e2['length'], edge_type='track')

            if node_is_mergeable(s):
                for a in ('xys', 'xyos'):
                    s[a].extend(v[a])
                s['length'] += v['length']
            else:
                for a in ('xys', 'xyos'):
                    v[a].extend(t[a])
                    t[a] = v[a][:]
                t['length'] += v['length']

            g.delete_edges([e1, e2])
            vertices_to_delete.append(v)
    g.delete_vertices(vertices_to_delete)


def add_resource_nodes(g):
    non_resources = g.vs.select(node_type_ne='resource')
    for v in non_resources:
        same_resource_nodes = non_resources.select(lambda vtx: not set(v['xys']).isdisjoint(set(vtx['xys'])))
        rsci, = set(same_resource_nodes['resource'])
        if rsci < 0:
            rsc = g.add_vertex(x=v['x'], y=v['y'], node_type='resource', xys=v['xys'], xyos=v['xyos'])
            rsci = rsc.index
            for srn in same_resource_nodes:
                assert set(srn['xys']) == set(v['xys']), str(v['xys']) + str(srn['xys'])
                srn['resource'] = rsci
                g.add_edge(rsc, srn, edge_type='resource')
                g.add_edge(srn, rsc, edge_type='resource')
    assert len(non_resources.select(resource_lt=0)) == 0


def find_edges_that_share_resource(g):
    return [(i,j) for i in range(len(g.es)) for j in range(len(g.es)) if g.es[i]['pos'] == g.es[j]['pos'] and j>i]

def add_target_nodes(g,env):
    for i,a in enumerate(env.agents):  
        tx,ty=a.target
        vs = g.vs.select(lambda v: (tx, ty) in v['xys'], node_type_ne='resource')
        assert len(vs) == 2
        rsc, = g.vs.select(lambda v: (tx, ty) in v['xys'], node_type='resource')
        for v in vs:
            assert v['resource'] == rsc.index
            xyi = v['xys'].index((tx, ty))
            xybefore, xyafter = v['xys'][:xyi], v['xys'][xyi+1:]
            e1, e2 = v.in_edges()[0], v.out_edges()[0]
            s, t = e1.source_vertex, e2.target_vertex
   
def add_agent_nodes(g,env):
    for i,a in enumerate(env.agents):
        if not a.position is None: 
            x,y=a.position
            g.add_vertex(x=x,y=y,pos_of_agent=i)
        elif not a.initial_position is None:
            x,y=a.initial_position
            g.add_vertex(x=x,y=y,pos_of_agent=i)

        ### more information is needed here: velocity, malfunction, direction,old_position!



class TransitionGraph:
    def __init__(self, env):
        self.g = build_transition_graph(env)
        # merge_within_junction(self.g)  # when a train comes into a junction with only one exit, how it comes in determines what orientation it is, which is redundant, because it can only go into one place. this function merges those, but it's unclear whether this is useful right now.
        merge_linear_paths(self.g)
        add_resource_nodes(self.g)
        add_target_nodes(self.g,env)
        # add_agent_nodes(self.g,env)


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
    matches=len(g.vs.select(x=v['x'], y=v['y']))
    return matches>2

def is_dead_end(v):
    return len(v.successors())==1 and abs(v['o'] - v.successors()[0]['o'])==2

