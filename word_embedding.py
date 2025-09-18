import torch.nn as nn
from preprocessing import *


class CNN_word_embedding(nn.Module):
    def __init__(self, char_vocab, d=10, D=6, width=2):
        super(CNN_word_embedding, self).__init__()

        self.d = d
        self.D = D
        self.width = width

        self.embedding = nn.Embedding(len(char_vocab), d)
        self.convolution = nn.Conv1d(in_channels=d,
                                     out_channels=D,  # number of filters
                                     kernel_size=width,  # width of each filter
                                     bias=True)
        self.maxpool = nn.MaxPool1d(kernel_size = char_vocab.L_token - width + 1)
    def forward(self, preprocessed_sentences):
        '''
        1. Tokend in preprocessed sentences are represented as OHE vectors
        OHE preprocessed data ->  (Nb, L_sentence, L_token, V)
        2. OHE tokens go through a learnable linear layer
        and became dense d-dimensional tensors
        OHE + linear layer ->  (Nb, L_sentence, L_token, d) d<<V
        3. D different (learnable) filters of shape (d, width)
        slide through embedded data + Max pool
        1D convolution + Max pool -> (Nb, L_sentence, D)

        :param preprocessed_sentences: (Nb, L_sentence, L_token) tensor,
            batch of OHE sentences char by char
        :return: (Nb, L_sentence, D) tensor,
            batch of embedded sentences word by word
        '''
        Nb, L_sentence, L_token = preprocessed_sentences.shape

        embedded_sentences = self.embedding(preprocessed_sentences)  # (Nb, L_sentence, L_token, d)

        embedded_sentences = embedded_sentences.transpose(2, 3)  # (Nb, L_sen, d, L_token)
        embedded_sentences = embedded_sentences.reshape(Nb * L_sentence, self.d, L_token)  #  (Nb * L_sen, d, L_token)
        output = self.convolution(embedded_sentences) # (Nb * L_sentence, D, L_token - width + 1)
        output = self.maxpool(output)  #  (Nb * L_sentence, D)
        output = output.reshape(Nb, L_sentence, self.D)

        return output

if __name__ == '__main__':
    # Data
    sentences = training_sentences

    # Build vocabulary
    char_vocab = Char_vocabulary(sentences)

    # Preprocessing data
    preprocessed_sentences = char_preprocess_sentences(sentences, char_vocab)  # (Nb, L_sentence, L_token)

    # Forward pass
    word_embedding = CNN_word_embedding(char_vocab)
    embedded_sentences = word_embedding(preprocessed_sentences)  # (Nb, L_sen, D)