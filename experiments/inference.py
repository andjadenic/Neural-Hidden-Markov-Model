import torch

from pythonProject1.models.word_embedding import *
from pythonProject1.models.tag_embedding import *
from pythonProject1.models.nhmm import *
from pythonProject1.data.data import *
from pythonProject1.data.collate_fn import *
from torch.utils.data import DataLoader
from pythonProject1.experiments.training import Small_WSJ_Dataset


if __name__ == "__main__":
    # Make vocabularies, testing Dataset and DataLoader

    char_vocab = Char_vocabulary(small_training_sentences)
    char_V = len(char_vocab)
    
    tag_vocab = Tag_vocabulary(small_training_tags)
    K = len(tag_vocab)
    
    word_vocab = Word_vocabulary(small_training_sentences)
    word_V = len(word_vocab)
    
    testing_dataset = Small_WSJ_Dataset(raw_x=small_training_sentences,
                                        raw_y=small_training_tags,
                                        word_vocab=word_vocab,
                                        tag_vocab=tag_vocab)
    
    '''testing_dataloader = DataLoader(testing_dataset,
                                    batch_size=Nb,
                                    shuffle=False,
                                    collate_fn=collate_fn)'''
    
    # Load the trained models and set them for evaluation
    word_embedding_model = CNN_word_embedding(char_vocab, d, D, width)
    tag_embedding_model = Basic_tag_embedding(K, D)
    nhmm_model = NHMM(d, width, D, word_V, char_V, K, num_layers=num_layers, dropout=0)
    
    word_embedding_model.eval()
    tag_embedding_model.eval()
    nhmm_model.eval()
    
    checkpoint = torch.load(r'D:\Faks\MASTER\PyTorch\Neural Hidden Markov Model\pythonProject1\models\models.pth')
    
    word_embedding_model.load_state_dict(checkpoint['word_embedding_model'])
    print('Pretrained word embedding was successfully loaded.')
    tag_embedding_model.load_state_dict(checkpoint['tag_embedding_model'])
    print('Pretrained tag embedding was successfully loaded.')
    nhmm_model.load_state_dict(checkpoint['nhmm_model'])
    print('Pretrained Neural Hidden Markov Model was successfully loaded.')


    # Create lookup matrix of embedded words
    W = char_matrix(word_vocab, char_vocab)  # (word_V, L_token)
    embedded_W = word_embedding_model(W)  # (word_V, D)

    # Create lookup matrix of embedded tags
    embedded_T = embedd_tag_matrix(tag_vocab, tag_embedding_model).reshape(K, D)  # (K, D)

    # Define trained emission matrix B
    B = embedded_T @ embedded_W.T

    # Define observation sequence
    test_sentence, test_targets = testing_dataset[0]  # (L_sentence,), (L_sentence,)
    L_sentence = test_sentence.shape[0]
    test_embedded_sentence = embedded_W[test_sentence]  # (L_sentence, D)

    # Define trained transition matrices A_t
    h, _ = nhmm_model.lstm(test_embedded_sentence[:-1, :])  # (L_sentence - 1, D)
    As = nhmm_model.transition_fc(h)  # (L_sentence - 1, K^2)
    As = As.reshape(-1, K, K)

    # Define initial probabilities pi
    pi = torch.ones((K, ), dtype=torch.double) / K  # uniform distribution

    # Decode observation sequence using Viterbi algorithm
    # input: As, B, pi, test_sentence
    # output: Likelihood and the best route
    # v[t, k] = P{w0, w1, ..., w_t, t_0, ..., t_t-1, t_t=k}
    # p[t, k] is back pointer to the state q_{t-1} that is the most probable
    '''v = torch.ones((L_sentence, K), dtype=torch.double)
    b = torch.ones((L_sentence, K), dtype=torch.int)

    v[0, :] = pi * B[]
    for t in range(1, L_sentence):
        v[t, :] = 
        b[t, :] = 
    https://web.stanford.edu/~jurafsky/slp3/A.pdf#page=8.13    
    '''


