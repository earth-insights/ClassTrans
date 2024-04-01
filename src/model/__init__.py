import torch
import torch.nn.functional as F
from torch import nn

from .pspnet import PSPNet
from .unet import UNet
from .deeplabv3p import DeepLabV3Plus
from .upernet import UPerNet


def get_model(args) -> nn.Module:
    if args.model_name == 'PSPNet':
        return PSPNet(args)
    elif args.model_name == 'UNet':
        return UNet(args)
    elif args.model_name == 'DeepLabV3Plus':
        return DeepLabV3Plus(args)
    elif args.model_name == 'UPerNet':
        return UPerNet(args)
    else:
        return NotImplementedError