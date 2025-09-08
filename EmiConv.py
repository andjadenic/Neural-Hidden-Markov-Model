import torch
import torch.nn as nn
import torch.nn.functional as F

import factory  # assumes you have a factory.build_cnn defined


class EmiConv(nn.Module):
    def __init__(self, word2char, nvars, feature_maps, kernels, charsize, hidsize):
        """
        word2char: Tensor (V, maxchars) mapping each word to sequence of char indices
        nvars: number of latent states (K)
        feature_maps: list of ints, number of filters for each kernel size
        kernels: list of ints, kernel sizes
        charsize: character embedding dim
        hidsize: hidden size for state embeddings
        """
        super().__init__()
        K = nvars
        self.word2char = word2char  # (V, maxchars)
        H = hidsize
        V, maxchars = word2char.size()
        nchars = int(word2char.max().item() + 1)  # +1 because indices are 0-based in PyTorch

        # Character CNN (provided by your factory)
        self.char_cnn = factory.build_cnn(feature_maps, kernels, charsize, hidsize, nchars, maxchars)

        # State embeddings
        self.state_emb = nn.Sequential(
            nn.Embedding(K, H),
            nn.ReLU()
        )

        # Shared bias
        self.bias = nn.Linear(1, V, bias=False)

        # Final log-softmax over vocabulary
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # Precomputation cache
        self._cache = None

    def forward(self, state_ids, word2char=None, bias_input=None):
        """
        state_ids: LongTensor (K,) state indices
        word2char: optional (V, maxchars) tensor, defaults to self.word2char
        bias_input: optional (K, 1) tensor, defaults to ones
        Returns: (K, V) log-probabilities
        """
        if word2char is None:
            word2char = self.word2char
        if bias_input is None:
            bias_input = torch.ones(state_ids.size(0), 1, device=state_ids.device)

        # State embeddings
        state_vecs = self.state_emb(state_ids)         # (K, H)

        # Char CNN
        char_vecs = self.char_cnn(word2char)           # (V, H)

        # Bilinear product (like nn.MM(false,true))
        emi0b = torch.matmul(state_vecs, char_vecs.T)  # (K, V)

        # Add bias
        bias_out = self.bias(bias_input)               # (K, V)
        emi = emi0b + bias_out

        return self.log_softmax(emi)

    def precompute(self):
        """Compute and cache the emission matrix (K, V) log-probs for all states/words."""
        with torch.no_grad():
            state_ids = torch.arange(self.state_emb[0].num_embeddings, device=self.word2char.device)
            bias_input = torch.ones(len(state_ids), 1, device=self.word2char.device)
            self._cache = self.forward(state_ids, self.word2char, bias_input)

    def log_prob(self, input):
        """
        input: LongTensor (N, T) of word indices
        Returns: log-probabilities (N, T, K)
        """
        N, T = input.size()
        if self._cache is None:
            self.precompute()
        logp = self._cache  # (K, V)

        # Select emissions for input words
        out = logp[:, input.view(-1)]  # (K, N*T)
        out = out.view(logp.size(0), N, T)  # (K, N, T)
        return out.permute(1, 2, 0)   # (N, T, K)
