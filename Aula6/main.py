import pyglet
from pyglet.window import mouse, key
from pyglet.gl import *

import shapes
from dataset import Dataset
from perceptron import Perceptron

dim = 600
delta = 50 ## 50 represents a scale of one

#creates window
window = pyglet.window.Window(dim, dim, caption='Linear Regression', resizable=True)
pyglet.gl.glClearColor(1.0, 1.0, 1.0, 1.0)

## Creating the dataset
dataset = Dataset(30)
## Creating the perceptron
perceptron = Perceptron(1e-3)

@window.event
def on_key_press(symbol, modifiers):
    global dataset, perceptron
    if symbol is key.W:
        for x, y in dataset:
            perceptron.train(x, y)
    elif symbol is key.D:
        dataset = Dataset(30)
    elif symbol is key.P:
        perceptron = Perceptron(1e-3)
    elif symbol is key.L:
        perceptron.lr *= 1.5
    elif symbol is key.M:
        perceptron.lr *= 0.9

def train(dt):
    for x,y in dataset:
        perceptron.train(x, y)

@window.event
def on_draw():
    window.clear()

    glPushMatrix()
    glViewport(0, 0, window.width, window.height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(200, dim, 200, dim, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    ##
    shapes.cartesian(dim, dim, delta)
    shapes.grid(dim, dim, delta)
    dataset.show(6 * delta, dim/2.)
    perceptron.show(6 * delta, dim/2.)
    ##
    glPopMatrix()

pyglet.clock.schedule_interval(train, 0.01)
pyglet.app.run()