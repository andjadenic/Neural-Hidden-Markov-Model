from pythonProject1.data.preprocessing import *
import torch.nn as nn

class Basic_tag_embedding(nn.Module):
    def __init__(self, tag_vocab, D):
        super(Basic_tag_embedding, self).__init__()

        self.D = D
        self.tag_vocab = tag_vocab

        self.embedding = nn.Embedding(len(tag_vocab), D)
        self.relu = nn.ReLU()

    def forward(self, preprocessed_tags):
        '''
        processed_tags: (Nb, L_sentence) or (L_sentence, ) tensor

        returns: (Nb, L_sentence, D) or (L_sentence, D) tensor
        '''
        output = self.embedding(preprocessed_tags)
        output = self.relu(output)
        return output