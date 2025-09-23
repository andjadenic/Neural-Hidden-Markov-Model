from pythonProject1.models.word_embedding import *
from pythonProject1.models.tag_embedding import *
from pythonProject1.models.nhmm import *
from pythonProject1.data.data import *
import itertools
from pythonProject1.data.collate_fn import *
from torch.utils.data import Dataset, DataLoader


class Small_WSJ_Dataset(Dataset):
    def __init__(self, raw_x, raw_y, word_vocab, tag_vocab):
        """
        raw_x: list of sentences represented as strings
        raw_y: list of lists of tags represented as strings
        """
        self.raw_x = raw_x  # Raw sentences
        self.raw_y = raw_y  # Raw tags

        self.x = [preprocess_sentence(sentence, word_vocab) for sentence in self.raw_x]  # list of tensors
        self.y = [preprocess_target(raw_target, tag_vocab) for raw_target in self.raw_y]  # list of words

        print('The Dataset was successfully created.')

    def __len__(self):
        return len(self.raw_x)

    def __getitem__(self, id):
        return (self.x[id], self.y[id])  # (L_sentence, ), (L_sentence, )


if __name__ == "__main__":
    # Build char vocabulary
    char_vocab = Char_vocabulary(small_training_sentences)
    char_V = len(char_vocab)

    # Build tag vocabulary
    tag_vocab = Tag_vocabulary(small_training_tags)
    K = len(tag_vocab)

    # Build word vocabulary
    word_vocab = Word_vocabulary(small_training_sentences)
    word_V = len(word_vocab)

    # Make words lookup table
    W = char_matrix(word_vocab, char_vocab)  # (word_V, L_token) tensor
    # W has numericized words (on char level) as rows

    # Wrap data in a Dataset and DataLoader
    training_dataset = Small_WSJ_Dataset(raw_x=small_training_sentences,
                                         raw_y=small_training_tags,
                                         word_vocab=word_vocab,
                                         tag_vocab=tag_vocab)

    training_dataloader = DataLoader(training_dataset,
                                     batch_size=Nb,
                                     shuffle=False,
                                     collate_fn=collate_fn)

    # Initialize word embedding model, tag embedding model and Neural Hidden Markov model
    word_embedding_model = CNN_word_embedding(char_vocab, d, D, width)
    tag_embedding_model = Basic_tag_embedding(K, D)
    nhmm_model = NHMM(d, width, D, word_V, char_V, K, num_layers=num_layers, dropout=0)

    # Initialize loss and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        itertools.chain(word_embedding_model.parameters(),
                        tag_embedding_model.parameters(),
                        nhmm_model.parameters()),
        lr=lr
    )

    # Train the model
    print('Тraining of models begins.')
    for epoch in range(N_epochs):
        word_embedding_model.train()
        tag_embedding_model.train()
        nhmm_model.train()

        for b, (batch_sentences, batch_tags) in enumerate(training_dataloader):
            # batch_sentence : (Nb, L_sentence)
            # batch_tag : (Nb, L_sentence)
            print('Batch: ', b)
            L_sentence = batch_sentences.shape[1]

            # embedded_W is a matrix that has embedded words 0, 1, ...word_V as  rows
            embedded_W = word_embedding_model(W)  # (word_V, D)

            # batch_tag : (Nb, L_sentence)
            embedded_tags = tag_embedding_model(batch_tags)  # (Nb, L_sentence, D)

            transitions, emissions = nhmm_model(batch_sentences, batch_tags, embedded_tags, embedded_W)
            # transitions: (Nb * (L_sentence - 1), K)
            # emissions: # (Nb * L_sentence, word_V)

            # Calculate loss in current batch
            batch_loss = loss(transitions, batch_tags[:, 1:].reshape(Nb * (L_sentence - 1), )) + \
                         loss(emissions, batch_sentences.reshape(Nb * L_sentence, ))
            print('Batch loss = ', batch_loss.item())

            # backward
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
    print('Model training has been successfully completed.')

    # Save model parameters
    torch.save({
        'word_embedding_model': word_embedding_model.state_dict(),
        'tag_embedding_model': tag_embedding_model.state_dict(),
        'nhmm_model': nhmm_model.state_dict()
    }, r'D:\Faks\MASTER\PyTorch\Neural Hidden Markov Model\pythonProject1\models\models.pth')
    print('Word and tag embedding models and NHMM were successfully saved.')