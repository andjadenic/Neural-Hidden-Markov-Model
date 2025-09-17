from data import *
from preprocessing import *
import torch.nn as nn

class Basic_tag_embedding(nn.Module):
    def __init__(self, tags_vocab, D):
        super(Basic_tag_embedding, self).__init__()

        self.D = D
        self.tag_vocab = tags_vocab

        self.embedding = nn.Embedding(len(tags_vocab), D)
        self.relu = nn.ReLU()

    def forward(self, preprocessed_tags):
        '''
        processed_tags: (Nb, L_sentence) or (L_sentence, ) tensor

        returns: (Nb, L_sentence, D) or (L_sentence, D) tensor
        '''
        output = self.embedding(preprocessed_tags)
        output = self.relu(output)
        return output


if __name__ == "__main__":
    # Build tag vocabulary
    tag_vocab = Tag_vocabulary(training_tags)
    K = len(tag_vocab)
    tags_list = torch.arange(K, dtype=torch.int)

    # Preprocess tags
    preprocessed_annotations = preprocess_tags(training_tags, tag_vocab)

    # Build a tag embedding model
    tag_embedding = Basic_tag_embedding(tag_vocab, D=6)

    # Make embedded tags lookup table
    # tags_lookup_table[k, :] is embedded tag k
    K = len(tag_vocab)
    tags_list = torch.arange(K, dtype=torch.int)
    tags_lookup_table = tag_embedding(tags_list)  # (K, D) tensor