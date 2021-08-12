'''
Properly implemented ResNet for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.0017M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

__all__ = ['ResNet_byot', 'resnet20', 'resnet32_byot', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def conv1x1(in_planes, planes, stride=1):
    return nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)


def branchBottleNeck(channel_in, channel_out, kernel_size):
    middle_channel = channel_out // 4
    return nn.Sequential(
        nn.Conv2d(channel_in, middle_channel, kernel_size=1, stride=1),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),

        nn.Conv2d(middle_channel, middle_channel, kernel_size=kernel_size, stride=kernel_size),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),

        nn.Conv2d(middle_channel, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
    )



class ResNet_byot(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, use_norm=False):
        super(ResNet_byot, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.downsample1_1 = nn.Sequential(
            conv1x1(16 * block.expansion, 64 * block.expansion, stride=8),
            nn.BatchNorm2d(64* block.expansion),
        )
        self.bottleneck1_1 = branchBottleNeck(16 * block.expansion, 64 * block.expansion, kernel_size=8)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc1 = nn.Linear(64 * block.expansion, num_classes)

        self.downsample2_1 = nn.Sequential(
            conv1x1(32 * block.expansion, 64 * block.expansion, stride=4),
            nn.BatchNorm2d(64 * block.expansion),
        )
        self.bottleneck2_1 = branchBottleNeck(32 * block.expansion, 64 * block.expansion, kernel_size=4)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc2 = nn.Linear(64 * block.expansion, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        layer1_out = self.layer1(out)
        middle_output1 = self.bottleneck1_1(layer1_out)
        middle_output1 = self.avgpool1(middle_output1)
        middle_feat1 = middle_output1
        middle_output1 = torch.flatten(middle_output1, 1)
        middle_output1 = self.middle_fc1(middle_output1)

        layer2_out = self.layer2(layer1_out)
        middle_output2 = self.bottleneck2_1(layer2_out)
        middle_output2 = self.avgpool2(middle_output2)
        middle_feat2 = middle_output2
        middle_output2 = torch.flatten(middle_output2, 1)
        middle_output2 = self.middle_fc2(middle_output2)

        layer3_out = self.layer3(layer2_out)
        layer3_out = self.avgpool(layer3_out)
        final_feat = layer3_out
        layer3_out = torch.flatten(layer3_out, 1)
        layer3_out = self.linear(layer3_out)
        return layer3_out, middle_output1, middle_output2, final_feat, middle_feat1, middle_feat2

class ResNet_fe(nn.Module):

    def __init__(self, block, num_blocks):
        super(ResNet_fe, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return out

class Classifier(nn.Module):
    def __init__(self, feat_in, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(feat_in, num_classes)
        self.apply(_weights_init)

    def forward(self, x):
        x = self.fc(x)
        return x


def resnet20():
    return ResNet_s(BasicBlock, [3, 3, 3])

def resnet32_fe():
    return ResNet_fe(BasicBlock, [5, 5, 5])

def resnet32_byot(num_classes=10, use_norm=False):
    return ResNet_byot(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)


def resnet44():
    return ResNet_s(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet_s(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet_s(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet_s(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()