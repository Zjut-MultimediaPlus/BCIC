import sys
import torch
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import math
import os
import os.path as osp
import argparse
import json
import time



embedding = nn.Embedding(9221, 512)
input1 = Variable(torch.LongTensor(2, 4).fill_(1))
b = embedding(input1)
print(input1.size())
print(input1)
print(b.size())
bb = Variable(torch.LongTensor(10, 15).fill_(1))
ccx = embedding(bb)
print(ccx.size())