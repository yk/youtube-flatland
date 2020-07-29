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

import tkinter as tk
from PIL import ImageTk

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Hello World\n(click me)"
        self.hi_there["command"] = self.say_hi
        self.hi_there.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    def say_hi(self):
        print("hi there, everyone!")

class EnvTest(absltest.TestCase):
    
    def test_build_graph(self):


        manual_graph=False
        if manual_graph:
            specs = [
                    [(8, 0), (1,90 ), (1, 90), (6, 0), (1, 90), (8, 90),(0,0)],
                    [(1, 180), (0, 0), (0, 0), (1, 0), (0, 0), (1, 0),(0,0)],
                     [(8, 270), (1, 90), (1, 90), (2, 90), (1, 90), (2, 360),(0,0)],
                     [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0),(0,0)],
                     [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),(0,0)]
                     ]

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

        def render_env():
            render = rendertools.RenderTool(env, gl='PILSVG')
            render.render_env()
            bg_img = Image.fromarray(render.get_image()).convert('RGB')
            img = ImageTk.PhotoImage(bg_img)
            img.img = bg_img
            return img


        root = tk.Tk()
        bg_img = render_env()
        canvas = tk.Canvas(root, width = bg_img.img.size[0], height = bg_img.img.size[1])  
        canvas.pack()
        canvas_img = canvas.create_image(0, 0, anchor=tk.NW, image=bg_img)
        root.title('Yo')

        def keypress(event):
            nonlocal bg_img
            k = event.keysym
            action = dict(
                    Up=rail_env.RailEnvActions.MOVE_FORWARD, 
                    Down=rail_env.RailEnvActions.STOP_MOVING,
                    Left=rail_env.RailEnvActions.MOVE_LEFT,
                    Right=rail_env.RailEnvActions.MOVE_RIGHT,
                    space=rail_env.RailEnvActions.DO_NOTHING,
                    )[k]
            root.title('Loading...')
            env.step({0: action})
            bg_img = render_env()
            canvas.itemconfig(canvas_img, image=bg_img)
            root.title(f'Done: {k}')

        for k in ('<Left>', '<Right>', '<Up>', '<Down>', '<space>'):
            root.bind(k, keypress)
        root.mainloop()


if __name__ == '__main__':
    absltest.main()
