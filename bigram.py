import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32 # number of parallel independent sequences to run
chunk_size = 8 # max size of the chunks to run algo on
max_iters = 3000 # max times to run algo
eval_interval = 300
learning_rate = 1e-2
device ='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

