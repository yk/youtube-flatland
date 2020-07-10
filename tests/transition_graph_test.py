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


        manuel_graph=True
        if manuel_graph:
            specs = [[(8, 0), (1,90 ), (1, 90), (6, 0), (1, 90), (7, 90)],
                    [(1, 180), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0)],
                     [(8, 270), (1, 90), (1, 90), (2, 90), (1, 90), (7, 90)],
                     [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
                     [(1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]]

            rail_shape = np.array(specs).shape

            env = rail_env.RailEnv(
                width=rail_shape[1],
                height=rail_shape[0],
                rail_generator=rail_generators.rail_from_manual_specifications_generator(specs),
                number_of_agents=1
            )
        else:
            env = rail_env.RailEnv(
                width=25,
                height=15,
                rail_generator=rail_generators.sparse_rail_generator())


        observations, info = env.reset()



        #env = rail_env.RailEnv(
         #       width=5,
          #      height=5,
           #     rail_generator=rail_generators.complex_rail_generator())

        #obs, info = env.reset()
        #obs, info = env.reset()
        #obs, info = env.reset()
        
        tg = transition_graph.TransitionGraph(env)
        g = tg.g

        plt = igraph.Plot()
        layout = [(v['y']+np.random.randn()*0.15, v['x']+np.random.randn()*0.15) for v in g.vs]
    
        plt.add(g, layout=layout)
        plt.redraw()
        with tempfile.NamedTemporaryFile() as f:
            plt.save(f.name)
            graph_img = Image.open(f.name)

        render = rendertools.RenderTool(env, gl='PILSVG')
        render.render_env()
        bg_img = Image.fromarray(render.get_image())
        graph_img = graph_img.resize(bg_img.size)

        minx, maxx, miny, maxy = min(g.vs['x']), max(g.vs['x']), min(g.vs['y']), max(g.vs['y'])
        envh, envw = env.height, env.width

        w, h = bg_img.size
        graph_img = graph_img.resize((int((maxy - miny)/envw*w), int((maxx-minx)/envh*h)))
        bg_img = bg_img.resize((int(w/envw*(envw+1)), int(h/envh*(envh+1))))

        bg_img.paste(graph_img, (int(w/envw*(miny+.5)), int(h/envh*(minx+.5))), graph_img)
        bg_img.show()


if __name__ == '__main__':
    absltest.main()
