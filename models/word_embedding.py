import torch.nn as nn
from pythonProject1.utils.config import *


class CNN_word_embedding(nn.Module):
    def __init__(self, char_vocab, d=d, D=D, width=width):
        super(CNN_word_embedding, self).__init__()

        self.d = d
        self.D = D
        self.width = width
        self.char_V = len(char_vocab)
        self.L_token = char_vocab.L_token

        self.embedding = nn.Embedding(self.char_V, d)
        self.convolution = nn.Conv1d(in_channels=d,
                                     out_channels=D,  # number of filters
                                     kernel_size=width,  # width of each filter
                                     bias=True)
        self.maxpool = nn.MaxPool1d(kernel_size=self.L_token - self.width + 1)

        print('The word embedding was successfully created.')

    def forward(self, W):
        '''
        W: (word_V, L_token) tensor of numericized words (on char level) as rows

        :return: embedded_W (word_V, D) tensor
        of embedded sentences word by word

        1. Represent chars as OHE vectors
                result : (word_V, L_token, char_V) sparse tensor
        2. OHE chars go through a learnable linear layer
           and became dense d-dimensional tensors
                result : (word_V, L_token, d) tensor, where d << char_V
        3. Slide D different (learnable) filters (kernels) of shape (d, width)
           through embedded data + Max pool
           (1D convolution + Max pool)
                result: (word_V, D) tensor
        '''
        word_V, L_token = W.shape

        embedded_W = self.embedding(W)  # (word_V, L_token, d)
        embedded_W = embedded_W.transpose(1, 2)  # (word_V, d, L_token)

        embedded_W = self.convolution(embedded_W)  # (word_V, D, L_token - width + 1)

        embedded_W = self.maxpool(embedded_W)  # (word_V, D, 1)
        embedded_W = embedded_W.reshape(word_V, self.D)  # (word_V, D)
        return embedded_W