import kornia
import torch

traced_rgb2lab = torch.jit.trace(kornia.color.rgb_to_lab, (torch.randn((2, 3, 256, 256))))
traced_lab2rgb = torch.jit.trace(kornia.color.lab_to_rgb, (torch.randn((2, 3, 256, 256))))
torch.jit.save(traced_rgb2lab, "rgb2lab.pt")
torch.jit.save(traced_lab2rgb, "lab2rgb.pt")
