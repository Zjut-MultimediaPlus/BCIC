import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable

pretrained_model = models.vgg16(pretrained=True)

class Vgg16Feats(nn.Module):
  def __init__(self):
    super(Vgg16Feats, self).__init__()
    self.features_nopool = nn.Sequential(*list(pretrained_model.features.children())[:-1])
    self.features_pool = list(pretrained_model.features.children())[-1]
    self.classifier = nn.Sequential(*list(pretrained_model.classifier.children())[:-1])

  def forward(self, x):
    x = self.features_nopool(x)
    x_pool = self.features_pool(x)   ## (100, 512, 7, 7)
    # x_feat = x_pool.view(x_pool.size(0), -1)
    # y = self.classifier(x_feat)      ## (100, 4096)
    # return x_pool, y
    return x_pool


# feature = torch.nn.Sequential(*list(pretrained_model.children())[:])
# print(feature)
# print(pretrained_model._modules.keys())