from IPython import get_ipython


def is_running_in_jupyter_notebook():
    try:
        if 'IPKernelApp' not in get_ipython().config:  # Check if IPython kernel is running
            return False
    except (ImportError, AttributeError):
        return False

    return True
