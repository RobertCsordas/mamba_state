import mamba
import torch

B = 8
L = 128
dmodel = 512
n_layers = 4

cfg = mamba.MambaConfig(dmodel, n_layers)
model = mamba.Mamba(cfg).cuda()

x = torch.randn((B, L, dmodel), device='cuda')

print(model(x).shape)