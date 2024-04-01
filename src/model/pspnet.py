import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import segmentation_models_pytorch as smp


class PSPNet(nn.Module):
    """
    PSPNet模型用于语义分割任务。

    Args:
        args (dict): 包含模型参数的字典。

    Attributes:
        model (torch.nn.Module): 分割网络模型，用于获得特征图。
        bottleneck_dim (int): 瓶颈层维度，用于将特征图转换为输出结果。
        classifier (torch.nn.Conv2d): 用于将瓶颈特征映射到类别数的卷积层。

    Methods:
        freeze_bn(): 冻结Batch Normalization层。
        forward(x): PSPNet前向传播方法。
        extract_features(x): 提取特征的方法。
        classify(features, shape): 分类特征的方法。
    """

    def __init__(self, args):
        super(PSPNet, self).__init__()
        assert args.get(
            'num_classes_tr') is not None, 'Get the data loaders first'

        # 分割网络，获得Feature Map
        self.model = smp.PSPNet(
            encoder_name=args.encoder_name, classes=args.num_classes_tr)
        # 将Feature Map转换为输出的结果
        self.bottleneck_dim = self.model.decoder.conv[0].out_channels  # 512
        self.classifier = nn.Conv2d(
            self.bottleneck_dim, args.num_classes_tr, kernel_size=1)  # 512 -> 8

    def freeze_bn(self):
        """
        冻结Batch Normalization层的方法。
        """
        for m in self.modules():
            if not isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        """
        PSPNet的前向传播方法。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            logits (torch.Tensor): 分类结果的张量。
        """
        x_size = x.size()
        assert (x_size[2]) % 8 == 0 and (x_size[3]) % 8 == 0
        shape = (x_size[2], x_size[3])

        x = self.extract_features(x)
        logits = self.classify(x, shape)
        return logits

    def extract_features(self, x):
        """
        提取特征的方法。

        Args:
            x (torch.Tensor): 输入张量，即图像，形状为[batch_size, channel, width, height]。

        Returns:
            x (torch.Tensor): 提取的特征张量，即PSPNet输出的Feature Map，形状为[batch_size, bottleneck_dim, 128, 128]。
        """
        self.model.segmentation_head = torch.nn.Identity()
        x = self.model(x)
        return x

    def classify(self, features, shape):
        """
        分类特征的方法。

        Args:
            features (torch.Tensor): 特征张量。
            shape (tuple): 输出形状。

        Returns:
            x (torch.Tensor): 分类结果的张量。
        """
        x = self.classifier(features)
        x = F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x
