from data import *
from preprocessing import *
import torch.nn as nn

class basic_tag_embedding(nn.Module):
    def __init__(self, tags_vocab, D):
        super(basic_tag_embedding, self).__init__()

        self.D = D
        self.tag_vocab = tags_vocab

        self.embedding = nn.Embedding(len(tags_vocab), D)
        self.relu = nn.ReLU()

    def forward(self, preprocessed_tags):
        output = self.embedding(preprocessed_tags)
        output = self.relu(output)
        return output


if __name__ == "__main__":
    # Build tag vocabulary
    tags_vocab = Tags(training_tags)
    print(tags_vocab.tags)
    print(len(tags_vocab))

    # Preprocess tags
    preprocessed_annotations = preprocess_tags(training_tags, tags_vocab)

    tag_embedding = basic_tag_embedding(tags_vocab, D=6)
    lookup_table = tag_embedding(preprocessed_annotations)