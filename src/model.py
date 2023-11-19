from fastai.vision.all import *
from fastai.vision.models.unet import DynamicUnet
from fastai.vision.models import resnet18
import torch

m = resnet18(pretrained=True)
m = nn.Sequential(*list(m.children())[:-2])
arch = DynamicUnet(m, 2, (256, 256))
traced = torch.jit.trace(arch, (torch.randn((16, 3, 256, 256))))
torch.jit.save(traced, "resnet_unet.pt")