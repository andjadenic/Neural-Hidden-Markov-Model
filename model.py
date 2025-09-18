import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from preprocessing import *
from data import *
from tag_embedding import *
from word_embedding import *
from torch.utils.data import Dataset, DataLoader


class Small_WSJ_Dataset(Dataset):
    def __init__(self, raw_x, raw_y, char_vocab, tag_vocab):
        """
        raw_x: list of sentences represented as strings
        raw_y: list of lists of tags represented as strings
        """
        self.raw_x = raw_x  # Raw sentences
        self.raw_y = raw_y  # Raw tags

        self.x = [preprocess_sentence(sentence, char_vocab) for sentence in self.raw_x]  # list of tensors
        self.y = [preprocess_target(raw_target, tag_vocab) for raw_target in self.raw_y]

    def __len__(self):
        return len(self.raw_x)

    def __getitem__(self, id):
        return (self.x[id], self.y[id])


class NHMM(nn.Module):
    def __init__(self, D, V, K, num_layers=1, dropout=0):
        super(NHMM, self).__init__()

        # Model parameters
        self.D = D  # Size of (both) word and tag embedded space
        self.K = K  # Number of all possible hidden states (tags)
        self.V = V  # Word vocabulary size

        # Emission architecture:
        # Linear layer + softmax activation
        self.emission_fc = nn.Linear(self.D, self.V)

        # Transition architecture:
        # multiple layer LSTM + linear layer
        self.lstm = nn.LSTM(input_size=self.D,
                            hidden_size=self.D,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.transition_fc = nn.Linear(self.D, self.K ** 2)

    def forward(self, embedded_tags, x):
        '''
        embedded_tags: (K, D) tensor, list of all embedded tags
        x: (Nb, L_sequence, D) tensor,
            tag embeddings for Nb batches (sentences) in training data
        '''
        Nb, L_sequence, _ = x.shape

        # Emissions
        log_B = self.emission_fc(embedded_tags)  # log B(X = _ | Z = tag_embedding) (without softmax)

        # Transitions
        h, _ = self.lstm(embedded_tags)
        # embedded_tags: (Nb, L_sequence, D)
        # h = [h_1, h_2, h_3, ..., h_{L_sequence}]
        # h: (Nb, L, D)
        # I need: [h_0, h_1, h_2, ..., h_{L_sequence})

        A = self.transition_fc(h.reshape(self.Nb * self.L_sequence, self.D))  # (Nb * L, K^2)
        A = A.reshape(Nb, L_sequence, self.K, self.K)
        # A[b, t, :, :] is log transition probability matrix
        # A = F.softmax(A, dim=) # Normalization

        return log_B, A


if __name__ == "__main__":

    # Build char vocabulary
    char_vocab = Char_vocabulary(small_training_sentences)
    ch_V = len(char_vocab)

    # Build tag vocabulary
    tag_vocab = Tag_vocabulary(small_training_tags)
    K = len(tag_vocab)

    # Wrap data in a Dataset and DataLoader
    training_dataset = Small_WSJ_Dataset(raw_x=small_training_sentences,
                                         raw_y=small_training_tags,
                                         char_vocab=char_vocab,
                                         tag_vocab=tag_vocab)
    training_dataloader = DataLoader(training_dataset,
                                     batch_size=Nb,
                                     shuffle=False,
                                     collate_fn=collate_fn)

    # Initialize model, loss and optimizer
    # Build a Neural Hidden Markov Model
    '''nhmm_model = NHMM(D, V, K,
                      num_layers=num_layers,
                      dropout=dropout)'''

    # Try forward pass for single input

    # Train the model

    # Save model parameters

    # Load model parameters

    # Test predictions with loaded model
    
    '''# Build token (word) vocabulary
    word_vocab = Word_vocabulary(training_sentences)
    V = len(word_vocab)

    # Preprocess sentences
    char_preprocessed_sentences = char_preprocess_sentences(training_sentences, char_vocab)  # (Nb, L_sentence, L_token)

    # Build a tag embedding model
    tag_embedding = Basic_tag_embedding(tag_vocab, D)

    # Make embedded tags lookup table
    # tags_lookup_table[k, :] is embedded tag k
    tags_list = torch.arange(K, dtype=torch.int)
    tags_lookup_table = tag_embedding(tags_list)  # (K, D) tensor

    # Preprocess tags
    preprocessed_annotations = preprocess_tags(training_tags, tag_vocab)  # (Nb, L_sentence)

    # Build a word embedding (CNN)
    word_embedding = CNN_word_embedding(char_vocab, d, D, width)

    # to embed batch of preprocessed annotations just do word_embedding(prerocessed_annotations)
    '''