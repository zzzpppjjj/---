import torch.nn as nn
import torch
import math
#CBAM空间注意力加加通道注意力
#通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

#空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

#18、34层结构残差模块
class BasicBlock(nn.Module):
    expansion = 1 #代表网络中残差核个数是否发送变化

    # in_channel：输入特征矩阵的深度
    # out_channel：输出特征矩阵的深度
    #downsample：代表虚线残差结构
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,attention_module=None):
        super(BasicBlock, self).__init__()
        #注意力机制模块
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x #identity为捷径分支上的输出
        if self.downsample is not None: #判断是否有捷径分支
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity#加上捷径分支上的输出
        out = self.relu(out)
        return out
##18、34层结构残差模块+CBAM
class BasicBlockCbam(nn.Module):
    expansion = 1 #代表网络中残差核个数是否发送变化

    # in_channel：输入特征矩阵的深度
    # out_channel：输出特征矩阵的深度
    #downsample：代表虚线残差结构
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,attention_module=None):
        super(BasicBlockCbam, self).__init__()

        #注意力机制模块
        self.attention_module = attention_module

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

        self.stride = stride
        #CBAM
        self.spatial_atten = SpatialAttention()
        self.channel_atten = ChannelAttention(out_channel * self.expansion)


    def forward(self, x):
        identity = x #identity为捷径分支上的输出
        if self.downsample is not None: #判断是否有捷径分支
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # attention
        atten = self.channel_atten(out) * out
        atten = self.spatial_atten(atten) * atten

        if self.downsample is not None:
            identity = self.downsample(x)
        atten += identity#加上捷径分支上的输出
        out = self.relu(atten)
        return out

#18、34层结构残差模块+深度可分离卷积
class BlockDeepPoint(nn.Module):
    expansion = 1 #代表网络中残差核个数是否发送变化

    # in_channel：输入特征矩阵的深度
    # out_channel：输出特征矩阵的深度
    #downsample：代表虚线残差结构
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,attention_module=None):
        super(BlockDeepPoint, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv_dw = nn.Conv2d(in_channels=in_channel, out_channels=in_channel,
                               kernel_size=3, stride=stride, padding=1,groups=in_channel, bias=False)
        self.conv_point = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x #identity为捷径分支上的输出
        if self.downsample is not None: #判断是否有捷径分支
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv_dw(out)
        out = self.conv_point(out)
        out = self.relu(out)

        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity#加上捷径分支上的输出
        out = self.relu(out)
        return out
##18、34层结构残差模块+深度可分离卷积+CBAM
class BlockDeepPointCbam(nn.Module):
    expansion = 1 #代表网络中残差核个数是否发送变化

    # in_channel：输入特征矩阵的深度
    # out_channel：输出特征矩阵的深度
    #downsample：代表虚线残差结构
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,attention_module=None):
        super(BlockDeepPointCbam, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel)

        self.conv_dw = nn.Conv2d(in_channels=in_channel, out_channels=in_channel,
                               kernel_size=3, stride=stride, padding=1,groups=in_channel, bias=False)
        self.conv_point = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        self.stride = stride
        #CBAM
        self.spatial_atten = SpatialAttention()
        self.channel_atten = ChannelAttention(out_channel * self.expansion)

    def forward(self, x):
        identity = x #identity为捷径分支上的输出
        if self.downsample is not None: #判断是否有捷径分支
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv_dw(out)
        out = self.conv_point(out)
        out = self.relu(out)

        # attention
        atten = self.channel_atten(out) * out
        atten = self.spatial_atten(atten) * atten

        if self.downsample is not None:
            identity = self.downsample(x)
        atten += identity#加上捷径分支上的输出
        out = self.relu(atten)
        return out

#block：残差结构
#blocks_num：残差结构数数目（为列表类型）
#num_classes=1000：训练集的分类个数
#include_top=True：
#groups=1：
# width_per_group=64：
class ResNet(nn.Module):

    def __init__(self,
                 basic_block,
                 deep_block,
                 basic_cbam_block,
                 deep_cbam_block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64 #卷积核个数

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)

        # self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=2,
        #                        padding=1, bias=False)
        # self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1,
        #                        padding=1, bias=False)
        # self.conv3 = nn.Conv2d(3, self.in_channel, kernel_size=3, stride=1,
        #                        padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(basic_cbam_block, 64, blocks_num[0])
        self.layer2 = self._make_layer(deep_cbam_block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(deep_cbam_block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(deep_cbam_block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * deep_cbam_block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #block：残差模块
    #channel：第一层卷积核个数
    #block_num：该层包含了多少残差结构
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        #18和34层会直接跳过这个
        if stride != 1 or self.in_channel != channel * block.expansion:
            #进入虚线分支，会将特征图翻4倍
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            #groups=self.groups,
                            #width_per_group=self.width_per_group
                            ))
        self.in_channel = channel * block.expansion
        #将实线的残差结构全部加入进去
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel))
                                #groups=self.groups,
                                #width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def resnet18_DW(num_classes=4, include_top=True):
   return ResNet(BasicBlock,BlockDeepPoint,BasicBlockCbam,BlockDeepPointCbam, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

