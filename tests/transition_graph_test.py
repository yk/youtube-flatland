#!/usr/bin/env python3

from absl.testing import absltest

from absl import flags, logging

import tempfile

import numpy as np
import igraph
from PIL import Image

from flatland.envs import rail_env
from flatland.envs import rail_generators



from flatland.utils import rendertools

from youtube_flatland.environment import transition_graph

class EnvTest(absltest.TestCase):
    
    def test_build_graph(self):
        env = rail_env.RailEnv(
                width=25,
                height=15,
                rail_generator=rail_generators.sparse_rail_generator())

        obs, info = env.reset()
        g = transition_graph.build_transition_graph(env)

        #delete two paths as an example:
        delete_stuff=True
        if delete_stuff: 
            path_1=transition_graph.get_linear_path(g,g.vs[93]) 
            path_2=transition_graph.get_linear_path(g,g.vs[25]) 

        vs_to_delete=transition_graph.join_linear_path(g,path_1)      
        vs_to_delete+=transition_graph.join_linear_path(g,path_2)   

        g.delete_vertices(vs_to_delete)

        plt = igraph.Plot()
        layout = [(v['y']+np.random.randn()*0.075, v['x']+np.random.randn()*0.075) for v in g.vs]

    
        plt.add(g, layout=layout)
        plt.redraw()
        with tempfile.NamedTemporaryFile() as f:
            plt.save(f.name)
            graph_img = Image.open(f.name)

        render = rendertools.RenderTool(env, gl='PILSVG')
        render.render_env()
        bg_img = Image.fromarray(render.get_image())
        graph_img = graph_img.resize(bg_img.size)
        h, w = bg_img.size
        bg_img = bg_img.resize((h//5*6, w//5*6))

        bg_img.paste(graph_img, (h//12, w//12), graph_img)
        bg_img.show()


if __name__ == '__main__':
    absltest.main()
