import mamba
import torch

B = 32
L = 128
dmodel = 512
n_layers = 1

cfg = mamba.MambaConfig(dmodel, n_layers)
model = mamba.Mamba(cfg).cuda().type(torch.float64)

x = torch.randn((B, L, dmodel), device='cuda', dtype=torch.float64)

full_out = model(x)

for split_points in [1,2,3,4,L//2,L-2,L-3,L-4,L-5]:
    part1, state = model(x[:, :split_points], {})
    part2, state = model(x[:, split_points:], state)

    assert part1.allclose(full_out[:, :split_points])
    assert part2.allclose(full_out[:, split_points:])
    print(f"OK {split_points}")
