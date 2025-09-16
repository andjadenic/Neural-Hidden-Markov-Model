import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class NHMM(nn.Module):
    def __init__(self, V):
        super(NHMM, self).__init__()

        # Emission architecture:
        # Linear layer + softmax activation
        self.emission_fc = nn.Linear(D, V)


    def forward(self, x):
        output = self.emission_fc(x)  # raw logits, no softmax
        return output


if __name__ == "__main__":
    model = NHMM(V=10)
    x = torch.tensor([
        [[.9, 1, 2, 3, 4, 5],
         [.9, 1, 2, 3, 4, 5]],

        [[1, 2, 3, 4, 5, 6],
         [1, 2, 3, 4, 5, 6]],

        [[1, 2, 3, 4, 5, 6],
         [1, 2, 3, 4, 5, 6]]
    ])  # embedded 3 sent that has 2 words each
    print(x.shape)  # (Nb, L_sentence, D)  3, 2, 6
    o = model.forward(x)
    print(o)
    print(o.shape)  # (Nb, L_sentence, V)  3, 2, 10
