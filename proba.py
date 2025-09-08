import torch


K = 3
prior = torch.rand(K)
print(prior)
log_prior = torch.log(prior)
print(log_prior)
log_prior -= torch.logsumexp(log_prior, dim=0)  # normalize
print(log_prior)