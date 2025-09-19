import torch.nn as nn

class LSTM_transition(nn.Module):
    def __init__(self, D, V, K, num_layers=1, dropout=0):
        super(LSTM_transition, self).__init__()

        # Model parameters
        self.D = D  # Size of (both) word and tag embedded space
        self.K = K  # Number of all possible hidden states (tags)
        self.V = V  # Word vocabulary size

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

        return A





if __name__ == "__main__":
    print('hi')