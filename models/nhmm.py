from pythonProject1.data.collate_fn import *
import torch.nn as nn


class NHMM(nn.Module):
    '''
    Wraper architecture for supervised Neural Hidden Markov model
    '''
    def __init__(self, d, width, D, word_V, ch_V, K, num_layers=1, dropout=0):
        super(NHMM, self).__init__()

        # Model parameters
        self.d = d
        self.width = width
        self.D = D  # Size of (both) word and tag embedded space
        self.K = K  # Number of all possible hidden states (tags)
        self.word_V = word_V  # Word vocabulary size
        self.ch_V = ch_V

        # Transition architecture:
        # multiple layer LSTM + linear layer
        self.lstm = nn.LSTM(input_size=self.D,
                            hidden_size=self.D,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.transition_fc = nn.Linear(self.D, self.K ** 2)

        print('The Neural Hidden Markov Model was successfully created.')


    def forward(self, batch_sentences, batch_tags, embedded_tags, embedded_W):
        '''
        batch_sentences: (Nb, L_sentence)
        batch_tag : (Nb, L_sentence)
        embedded_tags : (Nb, L_sentence, D)
        embedded_W : (word_V, D)

        return: (transitions, emissions)
            transitions: (Nb, L_sentence, K)
            emissions: (Nb, L_sentence, word_V)
        '''
        Nb, L_sentence = batch_tags.shape

        # Emissions
        emissions = embedded_W @ embedded_tags.reshape(Nb * L_sentence, self.D).T

        #  (word_V, D) @ (D, Nb * L_sentence) = (word_V, Nb * L_sentence)
        emissions = emissions.T  # (Nb * L_sentence, word_V)

        # Transitions
        embedded_sentences = embedded_W[batch_sentences]  # (Nb, L_sentence, D)
        h, _ = self.lstm(embedded_sentences[:, :-1, :])  # (Nb, L_sentence - 1, D)
        h = h.reshape(Nb * (L_sentence - 1), self.D)

        transitions = self.transition_fc(h)  # (Nb * (L_sentence - 1), K^2)
        transitions = transitions.reshape(Nb, L_sentence - 1, self.K, self.K)

        # Expand batch_tags to match the shape for gather
        batch_tags_expanded = batch_tags[:, :-1].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.K)  # (Nb, L_sentence - 1, 1, K)

        # Gather along the K dimension (dim=2)
        transitions = torch.gather(transitions, dim=2, index=batch_tags_expanded)
        transitions = transitions.squeeze(2)  # (Nb, L_sentence - 1, K)
        transitions = transitions.reshape(Nb * (L_sentence - 1), self.K)
        return transitions, emissions