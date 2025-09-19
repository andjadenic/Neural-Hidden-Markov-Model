import torch.nn.functional

from pythonProject1.data.data import small_training_sentences
from pythonProject1.data.preprocessing import *
from pythonProject1.config import *


class CNN_word_embedding(nn.Module):
    def __init__(self, char_vocab, d=d, D=D, width=width):
        super(CNN_word_embedding, self).__init__()

        self.d = d
        self.D = D
        self.width = width
        self.ch_V = len(char_vocab)

        self.embedding = nn.Embedding(self.ch_V, d)
        self.convolution = nn.Conv1d(in_channels=d,
                                     out_channels=D,  # number of filters
                                     kernel_size=width,  # width of each filter
                                     bias=True)

    def forward(self, preprocessed_sentences):
        '''
        :param preprocessed_sentences: (Nb, L_sentence) tensor

        :return: (Nb, L_sentence, D) tensor,
            batch of embedded sentences word by word
        1. Represent chars as OHE vectors
            result : (Nb, L_sentence, L_token, ch_V) tensor
        2. OHE chars go through a learnable linear layer
            and became dense d-dimensional tensors
            result : (Nb, L_sentence, L_token, d) tensor, where d << ch_V
        3. Slide D different (learnable) filters of shape (d, width)
            through embedded data + Max pool
            (1D convolution + Max pool)
            result: (Nb, L_sentence, D) tensor
        '''
        Nb, L_sentence, L_token = preprocessed_sentences.shape

        embedded_sentences = self.embedding(preprocessed_sentences)  # (Nb, L_sentence, L_token, d)

        embedded_sentences = embedded_sentences.transpose(2, 3)  # (Nb, L_sentence, d, L_token)
        embedded_sentences = embedded_sentences.reshape(Nb * L_sentence, self.d, L_token)
        output = self.convolution(embedded_sentences)  # (Nb * L_sentence, L_token - width + 1, D)

        maxpool = nn.MaxPool1d(kernel_size=L_token - self.width + 1)
        output = maxpool(output)  # (Nb * L_sentence, D, 1)
        output = output.reshape(Nb * L_sentence, self.D)  # (Nb * L_sentence, D)
        output = output.reshape(Nb, L_sentence, self.D)
        return output