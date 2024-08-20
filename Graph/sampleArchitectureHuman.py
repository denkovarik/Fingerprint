from classes.Nodes import *
from classes.Graph import Graph
import os
from IPython import get_ipython
from IPython.display import display, Image
from utils import *


graph = Graph()
graph.construct(inputShape=torch.Size([1, 3, 32, 32]))
graph.sampleArchitectureHuman()

image_path = graph.render()

if isRunningInJupyterNotebook():
    display(Image(filename='Graphs/enas_network_search_space_visualization.png'))
else:
    os.system(f'feh {image_path}')


