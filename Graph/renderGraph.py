from classes.Nodes import *
from classes.Graph import Graph
import os
from IPython import get_ipython
from IPython.display import display, Image
from utils import *


graph = Graph()
graph.construct()
#print(graph.graph)
graph.sample = [1, 1, 1, 1, 7, 1, 1, 0, 3, 1, 1, 1, 0, 0, 0]
image_path = graph.render()

if is_running_in_jupyter_notebook():
    display(Image(filename='Graphs/enas_network_search_space_visualization.png'))
else:
    os.system(f'feh {image_path}')

