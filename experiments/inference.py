from pythonProject1.models.word_embedding import *
from pythonProject1.models.tag_embedding import *
from pythonProject1.models.nhmm import *
from pythonProject1.data.data import *
from pythonProject1.data.collate_fn import *
from pythonProject1.experiments.training import Small_WSJ_Dataset


def Viterbi(test_sentence, nhmm_model, embedded_W, pi, B):
    L_sentence = test_sentence.shape[0]
    test_embedded_sentence = embedded_W[test_sentence]  # (L_sentence, D)

    # Define trained transition matrices A_t
    h, _ = nhmm_model.lstm(test_embedded_sentence[:-1, :])  # (L_sentence - 1, D)
    As = nhmm_model.transition_fc(h)  # (L_sentence - 1, K^2)
    As = As.reshape(-1, K, K)

    # Decode observation sequence using Viterbi algorithm
    # input: As, B, pi, test_sentence
    # output: Likelihood and the best route
    # v[t, k] = P{w0, w1, ..., w_t, t_0, ..., t_t-1, t_t=k}
    # p[t, k] is back pointer to the state q_{t-1} that is the most probable
    v = - torch.ones((K, L_sentence), dtype=torch.double)
    b = - torch.ones((K, L_sentence), dtype=torch.int)

    v[:, 0] = pi * B[:, test_sentence[0].item()]
    for t in range(1, L_sentence):
        p = torch.ones((K, K))  # (from, into)
        for i in range(K):
            for j in range(K):
                p[i, j] = v[i, t - 1] * As[t - 1, i, j] * B[j, test_sentence[t].item()]
        for j in range(K):
            v[j, t] = torch.max(p[:, j])
            b[j, t] = torch.argmax(p[:, j])
    likelihoods = nn.functional.softmax(v[:, L_sentence - 1], dim=0)
    likelihood = likelihoods.max().item()

    best_path = []
    prev = likelihoods.argmax().item()
    best_path.append(prev)

    for t in reversed(range(0, L_sentence - 1)):
        prev = b[prev, t+1].item()
        best_path.append(prev)
    best_path.reverse()

    sentence = [word_vocab.id2word[word_id] for word_id in test_sentence.tolist()]
    prediction = [tag_vocab.id2tag[tag_id] for tag_id in best_path]
    target = [tag_vocab.id2tag[tag_id] for tag_id in test_targets.tolist()]

    print('Sentence: ', sentence)
    print('Prediction: ', prediction)
    print('Ground truth: ', target)

    return likelihood, best_path


if __name__ == "__main__":
    # Make vocabularies and testing Dataset
    char_vocab = Char_vocabulary(training_sentences)
    char_V = len(char_vocab)
    
    tag_vocab = Tag_vocabulary(training_tags)
    K = len(tag_vocab)
    
    word_vocab = Word_vocabulary(training_sentences)
    word_V = len(word_vocab)
    
    testing_dataset = Small_WSJ_Dataset(raw_x=training_sentences,
                                        raw_y=training_tags,
                                        word_vocab=word_vocab,
                                        tag_vocab=tag_vocab)
    
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
    B = embedded_T @ embedded_W.T  # (K, word_V)

    # Define initial probabilities pi
    pi = torch.ones((K,), dtype=torch.float)
    for tag in training_tags:
        tag_id = tag_vocab.tag2id[tag[0]]
        pi[tag_id] += 1
    pi /= len(training_tags)

    # Define test observation sequence
    test_sentence, test_targets = testing_dataset[55]  # (L_sentence,), (L_sentence,)

    # Do the inference using Viterbi algorithm
    likelihood, best_path = Viterbi(test_sentence, nhmm_model, embedded_W, pi, B)