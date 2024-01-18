from IPython import get_ipython
import os


def clearScreen():
    # For Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # For Mac and Linux (os.name: 'posix')
    else:
        _ = os.system('clear')


def isRunningInJupyterNotebook():
    try:
        if 'IPKernelApp' not in get_ipython().config:  # Check if IPython kernel is running
            return False
    except (ImportError, AttributeError):
        return False

    return True
