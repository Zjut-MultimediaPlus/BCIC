import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable

# pretrained_model = models.resnet101(pretrained=True)

class ResNet101(nn.Module):
  def __init__(self):
    super(ResNet101, self).__init__()
    self.net = models.resnet101(pretrained=True)
    #self.fc_2048 = nn.Sequential(*list(self.net.fc.children())[:-1])


  def forward(self, input):
      output_conv1 = self.net.conv1(input)
      output_bn1 = self.net.bn1(output_conv1)
      output_relu = self.net.relu(output_bn1)
      output_maxpool = self.net.maxpool(output_relu)
      output_layer1 = self.net.layer1(output_maxpool)
      output_layer2 = self.net.layer2(output_layer1)
      output_layer3 = self.net.layer3(output_layer2)
      output_layer4 = self.net.layer4(output_layer3)
      output_avgpool = self.net.avgpool(output_layer4)
      # output_avgpool = self.net.fc(output_avgpool)

      return output_avgpool


# feature = torch.nn.Sequential(*list(pretrained_model.children())[:])
# print(feature)
#
# print(pretrained_model._modules.keys())
#
# ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']