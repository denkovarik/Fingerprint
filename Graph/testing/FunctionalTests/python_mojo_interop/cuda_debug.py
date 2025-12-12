import sys
sys.path.insert(0, '/home/dennis/.local/lib/python3.10/site-packages')

import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.__config__.show())

