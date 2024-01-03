from classes.Nodes import *
from classes.Graph import Graph
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


graph = Graph()
graph.construct()
#print(graph.graph)
#graph.render()


image_path = 'enas_network_search_space_visualization.png'

#image = Image.open(image_path)
#image.show()

img = mpimg.imread(image_path)
imgplot = plt.imshow(img)
plt.show()
