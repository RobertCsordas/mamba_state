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
full_out.sum().backward()

grads = {n: p.grad.clone() for n, p in model.named_parameters()}

for split_points in [1,2,3,4,L//2,L-2,L-3,L-4,L-5]:
    for p in model.parameters():
        p.grad.zero_()

    part1, state = model(x[:, :split_points], {})
    part2, state = model(x[:, split_points:], state)

    assert part1.allclose(full_out[:, :split_points])
    assert part2.allclose(full_out[:, split_points:])
    print(f"Forward OK {split_points}")

    part1.sum().backward(retain_graph=True)
    part2.sum().backward()

    for n, p in model.named_parameters():
        assert grads[n].allclose(p.grad), f"Failed {n} {split_points}"
    
    print(f"Backward OK {split_points}")


for maxp in [L-1, 3, 2, 8]:
    splitpoints = torch.randint(1, maxp, (B,)).cuda()
    x1 = x.clone()
    for i in range(B):
        x1[i, splitpoints[i]:] = 0

    x2 = x.clone()
    for i in range(B):
        x2[i, :L-splitpoints[i]] = x[i, splitpoints[i]:]

    part1, state = model(x1, {}, lengths=splitpoints)
    part2, state = model(x2, state)

    for i in range(B):
        part1[i, splitpoints[i]:] = part2[i, :L-splitpoints[i]]

    assert part1.allclose(full_out)

    print(f"Splitpoint OK {maxp}")
