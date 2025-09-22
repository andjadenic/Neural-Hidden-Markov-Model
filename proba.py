import torch

# Example shapes
Nb, L, M, K = 2, 3, 4, 5
prediction = torch.randn(Nb, L, M, K)   # (Nb, L, M, K)
print(prediction, '\n')
t = torch.randint(0, M, (Nb, L,))       # (Nb, L)
print(t, '\n')

# Expand t to match the shape for gather
t_expanded = t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, K)  # (Nb, L, 1, K)

# Gather along the M dimension (dim=2)
result = torch.gather(prediction, dim=2, index=t_expanded)

# Remove the singleton dimension (M reduced away)
result = result.squeeze(2)  # (Nb, L, K)

print(result.shape)  # (Nb, L, K)
print(result)

