#from IPython import get_ipython
import os


def clearScreen():
    # For Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # For Mac and Linux (os.name: 'posix')
    else:
        _ = os.system('clear')


def ensureDirpathExists(dirpath):
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)


def ensureFilepathExists(filepath):
    dirpath = os.path.dirname(filepath)
    ensureDirpathExists(dirpath)


def isRunningInJupyterNotebook():
    try:
        if 'IPKernelApp' not in get_ipython().config:  # Check if IPython kernel is running
            return False
    except (ImportError, AttributeError):
        return False

    return True


def renderGraph(graph, renderFP):
    """
    Helper function for rendering and displaying the graph

    :param graph: Instance of the Graph class to render
    :param renderFP: Filepath to temp dir
    """
    imagePath = graph.render(renderFP)        
    if isRunningInJupyterNotebook():
        display(Image(filename=imagePath))
    else:
        os.system(f'feh {imagePath}')

    # Remove files used to render graph
    fp = imagePath[:imagePath.rfind('.')]
    if os.path.exists(fp):
        os.remove(fp)
    if os.path.exists(imagePath):
        os.remove(imagePath)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo: 
        dict = pickle.load(fo, encoding='bytes')
    return dict

